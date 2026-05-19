from dataclasses import dataclass


@dataclass
class VLAConfig:
    model_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1

    # JEPA path
    jepa_in_dim: int = 1024
    jepa_state_dim: int = 32
    # Past 3 + current frame
    jepa_history: int = 4
    state_history: int = 16
    action_history: int = 16
    jepa_predictor_layers: int = 6

    # VL guidance path
    vl_in_dim: int = 2048
    vl_guidance_dropout: float = 0.1

    # DINO path
    dino_in_dim: int = 1024

    # Token counts for segmented fusion sequence
    num_jepa_action_tokens: int = 8
    num_dino_tokens: int = 4
    action_horizon: int = 8
    action_dim: int = 4

    # Action flow matching head
    flow_hidden_dim: int = 1024
    flow_num_layers: int = 4
    flow_time_embed_dim: int = 128
    flow_solver_steps: int = 24

    # RTC runtime
    rtc_execute_steps: int = 4
    rtc_execution_horizon: int = 4
    rtc_max_guidance_weight: float = 10.0

    # Loss weights
    lambda_jepa_latent: float = 0.05
