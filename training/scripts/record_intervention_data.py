"""基于 BC 基座的在线 DAgger 手柄干预采集。

冻结的 BC 基座先独立闭环执行；只有当 BC 进入失败/需要修复的状态时，
操作者才按 ``A`` 键进行干预。手柄输入作为 BC 动作上的专家增量；当前
episode 从第一帧开始完整记录。只有本 episode 发生过人工干预且最终成功
终止时，才把整条 episode 写入 intervention pool。

输出 ``intervention_data.npz`` 的主要字段：

``states``
    原始单帧观测；策略推理时按 BC 基座保存的 z-score 统计量归一化。
``actions``
    实际送入环境的动作。
``done``
    环境是否在该步终止；由于只保存成功恢复轨迹，成功样本的最后一帧为 1。

默认只写入发生过干预并最终成功的恢复轨迹；未按 A 键干预、失败、截断或
手动丢弃的轨迹不会进入 intervention pool。
"""

from __future__ import annotations

import argparse
from collections import deque
import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common.filter import LowPassFilter
from training.model.base.basic_model import load_base_policy
from training.envs.assemble_mujoco_env import AssembleMuJoCoEnv
from training.common import rl_utils
from training.common.checkpoint import has_weight, load_policy_cfg, load_state_dict
from training.common.observation_wrapper import ObsNormalizeWrapper


class InterventionRecorder:
    """让冻结 BC 基座运行，并记录 DAgger 式人工修正。"""

    COOLDOWN_SEC = 0.45
    DEADZONE = 0.1
    TOLERANCE = 5e-4

    # 手柄按键：A 干预开关，X 调速，Y 丢弃并随机重置，Back 丢弃当前 episode，
    # Start 在 --wait-for-start 模式下开始录制，Guide（button 8）退出。
    BUTTON_INTERVENTION = 0
    BUTTON_SPEED = 2
    BUTTON_RESET = 3
    BUTTON_ABORT = 6
    BUTTON_START = 7
    BUTTON_EXIT = 8

    _ARRAY_KEYS = ("states", "actions", "done")

    def __init__(
        self,
        xml_path: str,
        urdf_path: str,
        base_model_dir: str | Path,
        output_path: str | Path,
        frame_stack: int | None = None,
        intervention_scale: float = 1,
        device: str = "auto",
        wait_for_start: bool = False,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.intervention_scale = float(intervention_scale)
        if self.intervention_scale <= 0:
            raise ValueError("intervention_scale 必须 > 0")
        self.wait_for_start = bool(wait_for_start)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.button_cooldown = 0.0
        self.move_mode = 0
        self.intervention_active = False
        self.is_recording = not self.wait_for_start
        self._exit_requested = False
        self._step_counter = 0
        self._episode_serial = 0
        self._completed_episodes: list[dict[str, list]] = []
        self.current_episode = deque()
        self.episode_has_intervention = False

        base_model_dir = Path(base_model_dir)
        if not has_weight(base_model_dir, "policy_net"):
            raise FileNotFoundError(f"BC 基座权重不存在: {base_model_dir}")
        base_cfg = load_policy_cfg(base_model_dir) or {}
        if not base_cfg:
            raise RuntimeError(
                f"缺少 {base_model_dir / 'policy_cfg.json'}；当前基座配置必须显式提供，"
                "不能仅从权重可靠推断。"
            )
        inferred_stack = base_cfg.get("seq_len") or frame_stack or 10
        self.frame_stack = int(inferred_stack if frame_stack is None else frame_stack)
        if self.frame_stack < 1:
            raise ValueError("frame_stack 必须 >= 1")
        base_seq_len = base_cfg.get("seq_len")
        if base_seq_len is not None and int(base_seq_len) != self.frame_stack:
            raise ValueError(
                f"frame_stack={self.frame_stack} 与 BC 基座 seq_len={int(base_seq_len)} 不一致；"
                "请省略 --frame-stack 让脚本从基座配置推断，或使用相同长度。"
            )

        self.base_env = AssembleMuJoCoEnv(
            xml_path=xml_path,
            urdf_path=urdf_path,
            render_mode="human",
            is_use_force_control=True,
            admittance_m=6.0,
            admittance_j=0.6,
            admittance_k_t=np.array([1800, 1800, 3000]),
            admittance_k_r=np.array([12, 6, 20]),
            admittance_zeta_t=2.2,
            admittance_zeta_r=1.2,
            admittance_force_deadzone=0.1,
            admittance_torque_deadzone=0.005,
        )
        normalizer = ObsNormalizeWrapper.load_normalizer(base_model_dir)
        normalizer_path = base_model_dir / "obs_normalizer.npz"
        if normalizer is None and bool(base_cfg.get("obs_normalized", False)) and normalizer_path.exists():
            raw_obs_dim = int(base_cfg.get("raw_obs_dim") or 10)
            normalizer = rl_utils.RunningMeanStd(shape=(raw_obs_dim,))
            normalizer.load_normalizer(base_model_dir)
        self.env = rl_utils.wrap_frame_stack(
            ObsNormalizeWrapper(self.base_env, normalizer=normalizer),
            self.frame_stack, padding_type="reset",
        )

        self.device = self._resolve_device(device)
        self.base_policy = load_base_policy(base_cfg, self.device)
        self.base_policy.load_state_dict(
            load_state_dict(base_model_dir, "policy_net", map_location=self.device)
        )
        self.base_policy.eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.action_zero = np.zeros(
            int(base_cfg.get("action_dim", self.env.action_space.shape[0])),
            dtype=np.float32,
        )
        # 手柄人工修正使用低通滤波；BC 策略动作直接输出。
        self.correction_filter = LowPassFilter(cutoff_freq=5, dt=0.05)
        pygame.init()
        pygame.joystick.init()
        self.joystick = (
            pygame.joystick.Joystick(0)
            if pygame.joystick.get_count() > 0
            else None
        )
        if self.joystick is None:
            raise RuntimeError("未检测到手柄")
        self.joystick.init()

        self.obs, self.info = self.env.reset(options={"random_delta": self._random_delta()})
        gc.collect()
        gc.freeze()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        value = str(device).lower()
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 cuda，但当前 PyTorch 没有可用 CUDA")
        return torch.device(value)

    @staticmethod
    def _random_delta() -> np.ndarray:
        return np.array([
            np.random.uniform(-0.001, 0.001),
            np.random.uniform(-0.001, 0.001),
            np.random.uniform(-0.003, 0.003),
            np.random.uniform(-5, 5),
            np.random.uniform(-0.001, 0.001),
            np.random.uniform(-0.001, 0.001),
            np.random.uniform(-0.001, 0.001),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
        ], dtype=np.float64)

    def _reset_filter(self) -> None:
        self.correction_filter.reset()

    def _reset_environment(self) -> None:
        self.current_episode.clear()
        self.episode_has_intervention = False
        self.intervention_active = False
        self.move_mode = 0
        self._reset_filter()
        self.obs, self.info = self.env.reset(
            options={"random_delta": self._random_delta()}
        )

    def _edge_button(self, button: int, now: float) -> bool:
        if button >= self.joystick.get_numbuttons():
            return False
        if not self.joystick.get_button(button):
            return False
        if now - self.button_cooldown <= self.COOLDOWN_SEC:
            return False
        self.button_cooldown = now
        return True

    def _joystick_delta(self) -> np.ndarray:
        """把手柄输入映射为环境动作空间内的 correction delta。"""
        if not self.intervention_active:
            self._reset_filter()
            return self.action_zero.copy()

        def axis(index: int, default=0.0) -> float:
            if index >= self.joystick.get_numaxes():
                return default
            return float(self.joystick.get_axis(index))

        ax0, ax1, ax2 = axis(0), axis(1), axis(2)
        ax2 = (1.0 + ax2) / 2.0
        dx = -(abs(ax1) > self.DEADZONE) * ax1
        dy = -(abs(ax0) > self.DEADZONE) * ax0
        dz = (abs(ax2) > self.DEADZONE) * (2 * self._button(4) - 1) * ax2

        ax3, ax4, ax5 = axis(3), axis(4), axis(5)
        ax5 = (1.0 + ax5) / 2.0
        dr_x = (abs(ax3) > self.DEADZONE) * ax3
        dr_y = -(abs(ax4) > self.DEADZONE) * ax4
        dr_z = (abs(ax5) > self.DEADZONE) * (1 - 2 * self._button(5)) * ax5

        raw = np.concatenate(([dx, dy, dz], [dr_x, dr_y, dr_z])).astype(np.float32)
        raw *= self.intervention_scale / float(self.move_mode + 1)
        correction = self.correction_filter.filter(raw)
        if np.all(np.abs(correction) < self.TOLERANCE):
            return self.action_zero.copy()
        return np.asarray(correction, dtype=np.float32)

    def _button(self, button: int) -> int:
        if button >= self.joystick.get_numbuttons():
            return 0
        return int(self.joystick.get_button(button))

    def _handle_buttons(self) -> None:
        now = time.time()
        if self._edge_button(self.BUTTON_INTERVENTION, now):
            self.intervention_active = not self.intervention_active
            if self.intervention_active:
                self.episode_has_intervention = True
                self.logger.info("手柄干预 ON：当前 episode 将保存整条轨迹")
            else:
                self._reset_filter()
                self.logger.info("手柄干预 OFF：恢复 BC 基座自动执行，继续等待 episode 成功")

        if self._edge_button(self.BUTTON_SPEED, now):
            self.move_mode = (self.move_mode + 1) % 3
            self.logger.info("干预速度模式=%d（缩放 1/%d）", self.move_mode, self.move_mode + 1)

        if self._edge_button(self.BUTTON_RESET, now):
            self.logger.info("丢弃当前 episode 并随机重置")
            self._reset_environment()

        if self._edge_button(self.BUTTON_ABORT, now):
            self.logger.info("丢弃当前 episode 并随机重置")
            self._reset_environment()

        if self._edge_button(self.BUTTON_START, now) and self.wait_for_start:
            self.is_recording = not self.is_recording
            if self.is_recording:
                # 等待录制期间策略可能已经走过一段轨迹；开始录制时重新
                # reset，确保保存的 episode 从首帧开始。
                self._reset_environment()
                self.logger.info("开始保存当前 BC rollout")
            else:
                self.logger.info("暂停保存；策略仍继续运行")

        if self._button(self.BUTTON_EXIT):
            self._exit_requested = True

    def _bc_action(self, stacked_obs: np.ndarray) -> np.ndarray:
        policy_obs = np.asarray(stacked_obs, dtype=np.float32)
        with torch.no_grad():
            state = torch.as_tensor(
                policy_obs[None], dtype=torch.float32, device=self.device
            )
            bc_action = self.base_policy.mean_action(state)[0]
        return bc_action.detach().cpu().numpy().astype(np.float32)

    def _record_transition(
        self,
        raw_state: np.ndarray,
        executed_action: np.ndarray,
        done: bool,
    ) -> None:
        transition = {
            "states": np.asarray(raw_state, dtype=np.float32).copy(),
            "actions": executed_action.copy(),
            "done": float(done),
        }
        self.current_episode.append(transition)

    def _finish_episode(self, success: bool) -> None:
        steps = len(self.current_episode)
        if steps and success and self.episode_has_intervention:
            episode = {key: [] for key in self._ARRAY_KEYS}
            for transition in self.current_episode:
                for key in self._ARRAY_KEYS:
                    episode[key].append(transition[key])
            self._completed_episodes.append(episode)
            self.logger.info(
                "干预 episode 成功，保存完整轨迹：source_episode=%d steps=%d",
                self._episode_serial,
                steps,
            )
        elif steps:
            self.logger.info(
                "episode 丢弃：success=%s intervention=%s steps=%d",
                success,
                self.episode_has_intervention,
                steps,
            )
        self.current_episode.clear()
        self.episode_has_intervention = False
        self._episode_serial += 1

    @staticmethod
    def _episode_arrays(episode: dict[str, list]) -> dict[str, np.ndarray]:
        out = {}
        for key in InterventionRecorder._ARRAY_KEYS:
            values = episode[key]
            dtype = np.float32
            out[key] = np.asarray(values, dtype=dtype)
        return out

    def _append_to_file(self, episodes: list[dict[str, list]]) -> None:
        if not episodes:
            return
        new = {
            key: np.concatenate(
                [self._episode_arrays(ep)[key] for ep in episodes], axis=0
            )
            for key in self._ARRAY_KEYS
        }
        current_episode_count = len(episodes)
        current_transition_count = int(len(new["states"]))

        if self.output_path.exists():
            with np.load(self.output_path, allow_pickle=False) as old_file:
                missing = sorted(set(self._ARRAY_KEYS) - set(old_file.files))
                if missing:
                    raise ValueError(
                        f"已有数据文件 {self.output_path} 缺少字段 {missing}，"
                        "不能追加到当前数据池。"
                    )
                old = {key: np.asarray(old_file[key], dtype=np.float32)
                       for key in self._ARRAY_KEYS}
            for key in self._ARRAY_KEYS:
                new[key] = np.concatenate([old[key], new[key]], axis=0)

        tmp_path = self.output_path.with_name(self.output_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            **new,
        )
        os.replace(tmp_path, self.output_path)
        total_episode_count = int(np.sum(new["done"] > 0.5))
        total_transition_count = int(len(new["states"]))
        self.logger.info(
            "intervention pool 已更新：%s，本次保存 episodes=%d、transitions=%d；"
            "累计 episodes=%d、transitions=%d（states/actions/done）",
            self.output_path,
            current_episode_count,
            current_transition_count,
            total_episode_count,
            total_transition_count,
        )

    def _flush_completed(self) -> None:
        if self._completed_episodes:
            self._append_to_file(self._completed_episodes)
            self._completed_episodes.clear()

    def run(self) -> None:
        period = self.base_env.force_ctrl_steps * self.base_env.model.opt.timestep
        next_tick = time.perf_counter()
        self.logger.info(
            "在线干预采集启动：A=干预开关，X=调速，Y/Back=丢弃重置，Guide=退出；output=%s",
            self.output_path,
        )
        try:
            while not self._exit_requested:
                pygame.event.pump()
                self._handle_buttons()
                if self._exit_requested:
                    break

                stacked_obs = np.asarray(self.obs, dtype=np.float32)
                raw_state = self.base_env._get_observation()
                bc_action = self._bc_action(stacked_obs)
                base_action = bc_action
                correction = self._joystick_delta()
                requested_action = bc_action + correction
                clip_mask = np.abs(requested_action) > 1.0 + 1e-6 # 哪些动作超出了环境范围
                executed_action = np.clip(
                    requested_action,
                    self.env.action_space.low,
                    self.env.action_space.high,
                ).astype(np.float32)
                applied_delta = executed_action - base_action

                next_obs, reward, terminated, truncated, info = self.env.step(executed_action)
                done = bool(terminated or truncated)
                if self.is_recording:
                    self._record_transition(
                        raw_state,
                        executed_action,
                        done,
                    )

                self._step_counter += 1
                if self._step_counter % 10 == 0:
                    self.logger.info(
                        "step=%d mode=%s bc_max=%.3f human_req_max=%.3f "
                        "human_applied_max=%.3f clip=%.0f%% state=%s force=%s torque=%s",
                        self._step_counter,
                        "INTERVENTION" if self.intervention_active else "BC",
                        float(np.max(np.abs(base_action))),
                        float(np.max(np.abs(correction))),
                        float(np.max(np.abs(applied_delta))),
                        100.0 * float(np.mean(clip_mask)),
                        info.get("state"),
                        info.get("force"),
                        info.get("torque"),
                    )

                self.obs = next_obs
                self.info = info
                if done:
                    self._finish_episode(success=bool(info.get("success", False)))
                    self._flush_completed()
                    self.intervention_active = False
                    self._reset_filter()
                    self.obs, self.info = self.env.reset(
                        options={"random_delta": self._random_delta()}
                    )

                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        except KeyboardInterrupt:
            # 未走到成功终止的完整 episode 不写入数据池。
            self.current_episode.clear()
            self.episode_has_intervention = False
            self.logger.info("收到 KeyboardInterrupt，未完成 episode 已丢弃")
        finally:
            self.current_episode.clear()
            self.episode_has_intervention = False
            self._flush_completed()
            self.base_env.close()
            pygame.quit()


def parse_args() -> argparse.Namespace:
    root = rl_utils.find_project_root()
    parser = argparse.ArgumentParser(description="基于 BC 基座的在线 DAgger 手柄干预采集")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=root / "models" / "bc_model_ur5e",
        help="冻结 BC 基座 checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets" / "intervention_data_bc_dagger.npz",
        help="独立 intervention pool 输出文件",
    )
    parser.add_argument(
        "--frame-stack", type=int, default=None,
        help="时序长度；默认读取 BC 基座 policy_cfg.json 的 seq_len",
    )
    parser.add_argument("--intervention-scale", type=float, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wait-for-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    root = rl_utils.find_project_root()
    xml_path = str(root / "assets" / "mjcf" / "ur5e_assemble_sence.xml")
    urdf_path = str(root / "assets" / "urdf" / "ur5e_assemble.urdf")
    recorder = InterventionRecorder(
        xml_path=xml_path,
        urdf_path=urdf_path,
        base_model_dir=args.base_model,
        output_path=args.output,
        frame_stack=args.frame_stack,
        intervention_scale=args.intervention_scale,
        device=args.device,
        wait_for_start=args.wait_for_start,
    )
    recorder.run()


if __name__ == "__main__":
    main()
