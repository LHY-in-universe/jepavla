from __future__ import annotations

from dataclasses import dataclass

from .schema import SchedulerConfig


@dataclass(frozen=True)
class TickInfo:
    frame_idx: int
    action_step_idx: int
    jepa_tick_idx: int
    vl_tick_idx: int
    is_jepa_tick: bool
    is_vl_tick: bool


class ControlScheduler:
    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg

    @property
    def actions_per_jepa_tick(self) -> int:
        return self.cfg.actions_per_frame * self.cfg.jepa_refresh_frames

    @property
    def actions_per_vl_tick(self) -> int:
        return self.cfg.actions_per_frame * self.cfg.vl_refresh_frames

    def frame_to_action_step(self, frame_idx: int) -> int:
        return frame_idx * self.cfg.actions_per_frame

    def is_jepa_tick(self, frame_idx: int) -> bool:
        return frame_idx >= 0 and (frame_idx + 1) % self.cfg.jepa_refresh_frames == 0

    def is_vl_tick(self, frame_idx: int) -> bool:
        return frame_idx >= 0 and frame_idx % self.cfg.vl_refresh_frames == 0

    def tick_info(self, frame_idx: int) -> TickInfo:
        return TickInfo(
            frame_idx=frame_idx,
            action_step_idx=self.frame_to_action_step(frame_idx),
            jepa_tick_idx=frame_idx // self.cfg.jepa_refresh_frames,
            vl_tick_idx=frame_idx // self.cfg.vl_refresh_frames,
            is_jepa_tick=self.is_jepa_tick(frame_idx),
            is_vl_tick=self.is_vl_tick(frame_idx),
        )
