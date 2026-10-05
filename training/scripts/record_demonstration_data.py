"""固定 20Hz 连续手柄采集入口。

录制始终采用连续模式：即使摇杆回中也推进环境并记录零动作帧。
每个 episode 的中间缓冲使用 transition 字典，最终文件保持训练兼容的
``states/actions/dones`` 三个数组。

B 键可切换到自由控制模式。该模式不写入示教数据，只持续运行环境力控循环，
可在 MuJoCo viewer 中用鼠标拖动机械臂观察导纳和力控响应；再次按 B 键回到
手柄控制模式后才继续连续采集。
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from collections import deque

import numpy as np
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common.filter import LowPassFilter
from training.envs.assemble_mujoco_env import AssembleMuJoCoEnv
from training.common import rl_utils


class DataRecorder:
    COOLDOWN_SEC = 0.5
    DEADZONE = 0.1
    TOLERANCE = 5e-4
    MAX_FINAL_YAW_DEG = 10.0

    # B 切换手柄控制/自由控制，X 调速，Y 随机重置，Back 丢弃当前 episode，
    # Start 录制开关，Guide 退出。
    BUTTON_CONTROL_MODE = 1
    BUTTON_SPEED = 2
    BUTTON_RESET = 3
    BUTTON_ABORT = 6
    BUTTON_START = 7
    BUTTON_EXIT = 8

    def __init__(self, xml_path, urdf_path, save_data_path):
        self.xml_path = xml_path
        self.urdf_path = urdf_path
        self.save_data_path = str(save_data_path)
        self.button_cooldown = 0.0
        self.is_recording = False
        self.current_episode = deque()
        self.completed_episodes = []
        self.action_zero = np.zeros(6, dtype=np.float32)
        self.ctrl_mode = 0  # 0=手柄控制并可记录，1=自由控制/鼠标拖动 MuJoCo
        self.move_mode = 0
        self._continuous_log_ctr = 0
        self._free_log_ctr = 0

        self.random_delta = self._sample_random_delta()
        self.filter = LowPassFilter(cutoff_freq=5, dt=0.05)
        self.env = AssembleMuJoCoEnv(
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
        self.env.is_teleoperation = False
        self.env.reset(options={"random_delta": self.random_delta})

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        pygame.init()
        pygame.joystick.init()
        self.joystick = (
            pygame.joystick.Joystick(0)
            if pygame.joystick.get_count() > 0 else None
        )
        if self.joystick is None:
            raise RuntimeError("未检测到手柄")
        self.joystick.init()

        gc.collect()
        gc.freeze()

    @staticmethod
    def _sample_random_delta():
        # return np.array([
        #     np.random.uniform(-0.001, 0.001),
        #     np.random.uniform(-0.001, 0.001),
        #     np.random.uniform(-0.003, 0.003),
        #     np.random.uniform(-5, 5),
        #     np.random.uniform(-0.001, 0.001),
        #     np.random.uniform(-0.001, 0.001),
        #     np.random.uniform(-0.001, 0.001),
        #     np.random.uniform(-10, 10),
        #     np.random.uniform(-10, 10),
        #     np.random.uniform(-10, 10),
        # ], dtype=np.float64)

        return np.array([
                    np.random.uniform(-0.001, 0.001),
                    np.random.uniform(-0.001, 0.001),
                    np.random.uniform(-0.003, 0.003),
                    np.random.uniform(-5, 5),
                    np.random.uniform(-0.003, 0.003),
                    np.random.uniform(-0.003, 0.003),
                    np.random.uniform(-0.003, 0.003),
                    np.random.uniform(-15, 15),
                    np.random.uniform(-15, 15),
                    np.random.uniform(-30, 30),
                ], dtype=np.float64)

    def _reset_environment(self, discard_current=True):
        if discard_current:
            self.current_episode.clear()
        self.random_delta = self._sample_random_delta()
        self.filter.reset()
        self.move_mode = 0
        self._free_log_ctr = 0
        self.env.reset(options={"random_delta": self.random_delta})
        self.env.is_teleoperation = self.ctrl_mode == 1

    def _finish_episode(self, info, terminated, truncated):
        """只保留成功且最终 yaw 合格的完整 episode。"""
        if not self.current_episode:
            return
        success = bool(info.get("success", False))
        if terminated and success and not truncated:
            final_state = self.current_episode[-1]["state"]
            final_yaw_deg = float(np.degrees(final_state[3]))
            if abs(final_yaw_deg) <= self.MAX_FINAL_YAW_DEG:
                self.completed_episodes.append(self.current_episode)
                self.logger.info(
                    "当前 episode 完成：steps=%d，最终 yaw=%.1f°",
                    len(self.current_episode), final_yaw_deg,
                )
            else:
                self.logger.warning(
                    "最终 yaw=%.1f°超过阈值 %.0f°，丢弃当前 episode",
                    final_yaw_deg, self.MAX_FINAL_YAW_DEG,
                )
        else:
            self.logger.info(
                "当前 episode 未成功完成，丢弃 steps=%d",
                len(self.current_episode),
            )
        self.current_episode = deque()

    def _edge_button(self, button, now):
        if button >= self.joystick.get_numbuttons():
            return False
        if not self.joystick.get_button(button):
            return False
        if now - self.button_cooldown <= self.COOLDOWN_SEC:
            return False
        self.button_cooldown = now
        return True

    def _button(self, button):
        if button >= self.joystick.get_numbuttons():
            return 0
        return int(self.joystick.get_button(button))

    def _read_joystick_action(self):
        ax0, ax1, ax2 = (
            self.joystick.get_axis(0),
            self.joystick.get_axis(1),
            self.joystick.get_axis(2),
        )
        ax2 = (1 + ax2) / 2
        dx = -(abs(ax1) > self.DEADZONE) * ax1
        dy = -(abs(ax0) > self.DEADZONE) * ax0
        dz = (abs(ax2) > self.DEADZONE) * (2 * self._button(4) - 1) * ax2

        ax3, ax4, ax5 = (
            self.joystick.get_axis(3),
            self.joystick.get_axis(4),
            self.joystick.get_axis(5),
        )
        ax5 = (1 + ax5) / 2
        dr_x = (abs(ax3) > self.DEADZONE) * ax3
        dr_y = -(abs(ax4) > self.DEADZONE) * ax4
        dr_z = (abs(ax5) > self.DEADZONE) * (1 - 2 * self._button(5)) * ax5

        raw_action = np.concatenate(([dx, dy, dz], [dr_x, dr_y, dr_z])).astype(np.float32)
        raw_action /= float(self.move_mode + 1)
        action = self.filter.filter(raw_action)
        if np.all(np.abs(action) < self.TOLERANCE):
            return self.action_zero.copy()
        return np.asarray(action, dtype=np.float32)

    def _handle_buttons(self):
        now = time.time()
        if self._edge_button(self.BUTTON_CONTROL_MODE, now):
            self.ctrl_mode = (self.ctrl_mode + 1) % 2
            self.logger.info(
                "切换控制模式为 %d（0-手柄控制，1-自由控制/鼠标拖动）",
                self.ctrl_mode,
            )
            # 模式切换会改变控制语义，当前未完成 episode 不应跨模式保存。
            self._reset_environment(discard_current=True)
            self.env.is_teleoperation = self.ctrl_mode == 1

        if self._edge_button(self.BUTTON_SPEED, now):
            self.move_mode = (self.move_mode + 1) % 3
            self.logger.info(
                "移动速度模式=%d（缩放 1/%d）",
                self.move_mode, self.move_mode + 1,
            )
        if self._edge_button(self.BUTTON_RESET, now):
            self.logger.info("随机重置并丢弃当前 episode, 当前已录制 %d 个 episode", len(self.completed_episodes))
            self._reset_environment(discard_current=True)
        if self._edge_button(self.BUTTON_ABORT, now):
            self.logger.info("退出当前 episode 并丢弃数据, 当前已录制 %d 个 episode", len(self.completed_episodes))
            self._reset_environment(discard_current=True)
        if self._edge_button(self.BUTTON_START, now):
            if self.is_recording:
                self.is_recording = False
                self.current_episode.clear()
                self.logger.info("录制停止，未完成 episode 已丢弃, 当前已录制 %d 个 episode", len(self.completed_episodes))
            else:
                self._reset_environment(discard_current=True)
                self.is_recording = True
                self.logger.info("录制开始（固定 20Hz 连续模式）")

    def _free_control_step(self):
        """自由控制模式：持续运行力控，允许用 MuJoCo viewer 鼠标拖动机械臂。"""
        self.env.is_teleoperation = True
        _, info = self.env.mujoco_step()
        self._free_log_ctr += 1
        if self._free_log_ctr % 5 == 0:
            actual_pos = self.env.data.site_xpos[self.env.eef_site_id].copy()
            ctrl = self.env.ur5e_controller
            self.logger.info(
                "自由控制: actual_pos=%s | |F|=%.3fN | |T|=%.3fNm | "
                "admittance_dx=%s | state=%s",
                np.round(actual_pos, 4),
                float(np.linalg.norm(ctrl.calibrated_ft[:3])),
                float(np.linalg.norm(ctrl.calibrated_ft[3:])),
                np.round(ctrl.admittance_dx, 6),
                info.get("state"),
            )

    def _control(self, observation):
        if self.ctrl_mode == 1:
            self._free_control_step()
            return self._button(self.BUTTON_EXIT)

        self.env.is_teleoperation = False
        action = self._read_joystick_action()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._continuous_log_ctr += 1

        if self.is_recording:
            self.current_episode.append({
                "state": np.asarray(observation, dtype=np.float32).copy(),
                "action": action.copy(),
                "done": float(terminated or truncated),
            })
            if self._continuous_log_ctr % 5 == 0:
                self.logger.info(
                    "连续采集: action=%s，当前步数=%d，深度=%s，状态=%s，力=%s，力矩=%s",
                    action,
                    self.env.current_step,
                    info.get("depth"),
                    info.get("state"),
                    info.get("force"),
                    info.get("torque"),
                )
            if terminated or truncated:
                self._finish_episode(info, terminated, truncated)
                self._reset_environment(discard_current=False)
        return self._button(self.BUTTON_EXIT)

    @staticmethod
    def _episodes_to_arrays(episodes):
        states, actions, dones = [], [], []
        for episode in episodes:
            if not episode:
                continue
            states.append(np.stack([t["state"] for t in episode]).astype(np.float32))
            actions.append(np.stack([t["action"] for t in episode]).astype(np.float32))
            dones.append(np.asarray([t["done"] for t in episode], dtype=np.float32))
        if not states:
            return (
                np.zeros((0, 10), dtype=np.float32),
                np.zeros((0, 6), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return np.concatenate(states), np.concatenate(actions), np.concatenate(dones)

    def _save_data(self):
        new_states, new_actions, new_dones = self._episodes_to_arrays(self.completed_episodes)
        current_episode_count = len(self.completed_episodes)
        current_transition_count = int(len(new_states))
        if current_transition_count == 0:
            self.logger.info("本次没有成功完成的 episode，未保存")
            return

        if os.path.exists(self.save_data_path):
            with np.load(self.save_data_path, allow_pickle=False) as old:
                old_states = np.asarray(old["states"], dtype=np.float32)
                old_actions = np.asarray(old["actions"], dtype=np.float32)
                old_dones = np.asarray(
                    old["dones"] if "dones" in old.files else old["done"],
                    dtype=np.float32,
                )
            new_states = np.concatenate([old_states, new_states], axis=0)
            new_actions = np.concatenate([old_actions, new_actions], axis=0)
            new_dones = np.concatenate([old_dones, new_dones], axis=0)

        np.savez_compressed(
            self.save_data_path,
            states=new_states,
            actions=new_actions,
            dones=new_dones,
        )
        total_episode_count = int(np.sum(new_dones > 0.5))
        total_transition_count = int(len(new_states))
        self.logger.info(
            "数据已更新：本次保存 episodes=%d、transitions=%d；累计 episodes=%d、transitions=%d",
            current_episode_count,
            current_transition_count,
            total_episode_count,
            total_transition_count,
        )

    def recode_run(self):
        step_wall = self.env.force_ctrl_steps * self.env.model.opt.timestep
        t_next = time.perf_counter()
        try:
            while True:
                pygame.event.pump()
                self._handle_buttons()
                observation = self.env._get_observation()
                if self._control(observation):
                    self.logger.info("录制中断")
                    self.current_episode.clear()
                    break
                t_next += step_wall
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.perf_counter()
        except KeyboardInterrupt:
            self.current_episode.clear()
            self.logger.info("录制中断，未完成 episode 已丢弃")
        finally:
            self._save_data()
            self.env.close()
            pygame.quit()


if __name__ == "__main__":
    root_dir = rl_utils.find_project_root()
    save_data_dir = root_dir / "datasets"
    xml_path = str(root_dir / "assets/mjcf/ur5e_assemble_sence.xml")
    urdf_path = str(root_dir / "assets/urdf/ur5e_assemble.urdf")
    save_data_path = save_data_dir / "recorded_expert_data.npz"

    recorder = DataRecorder(xml_path, urdf_path, save_data_path)
    recorder.recode_run()
