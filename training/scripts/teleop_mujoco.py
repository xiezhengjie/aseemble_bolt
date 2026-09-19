import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目根目录：本文件位于 training/scripts/，向上两级即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import time
import mujoco
import mujoco.viewer
from algorithm.controllers import UR5eController
from utils.math_utils import *


class TeleopMujocoKeyboard:
    """UR5e键盘遥操作，控制末端位置平移
    力校准、导纳、OSC 均封装在 UR5eController 中
    """

    def __init__(
        self,
        xml_path=str(PROJECT_ROOT / "mjcf" / "ur5e_assemble_sence.xml"),
        urdf_path=str(PROJECT_ROOT / "urdf" / "ur5e_assemble.urdf"),
        pos_action_scale=0.004,       # 每次按键移动距离 4mm
        is_use_force_control=True,
        # --- 频率分离参数（与assemble_mujoco_env一致）---
        force_ctrl_steps=80,         # 每次按键总仿真步数（80ms@1ms）
        admittance_ratio=10,         # 导纳频率/策略频率 = 10
        verbose=True
    ):
        # MuJoCo模型加载
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 参数配置
        self.pos_action_scale = pos_action_scale

        # 频率分离参数
        self.force_ctrl_steps = force_ctrl_steps
        self.admittance_ratio = admittance_ratio
        self.admittance_sub_steps = max(2, force_ctrl_steps // admittance_ratio)  # 80//10=8

        # 末端site/body ID
        self.eef_body_id = self.model.body('tcp_link').id
        self.eef_site_id = self.model.site('tcp_site').id
        self.force_sensor_site_id = self.model.site('force_torque').id

        # 创建UR5eController（力校准+导纳+OSC）
        self.ur5e_controller = UR5eController(
            model=self.model,
            data=self.data,
            urdf_filename=urdf_path,
            # 导纳参数
            m=1, j=0.05, k_t=2000, k_r=10,
            zeta_t=2.2, zeta_r=1.2, b_t=None, b_r=None,
            arm_dof=6,
            f_0=np.zeros(6),
            is_use_force_control=is_use_force_control,
            # 力校准参数
            eef_body_id=self.eef_body_id,
            force_sensor_site_id=self.force_sensor_site_id,
            mass=self.model.body_mass[self.eef_body_id],
            cutoff_freq=50,
            force_tr=2,
            torque_tr=0.1,
            # 导纳死区
            force_deadzone=2.0,
            torque_deadzone=0.05,
            admittance_dt=self.model.opt.timestep,
            verbose=True
        )

        # 键盘按键状态
        self.key_pressed = None
        self.last_key = None

        # 初始化：复位到初始位姿
        self._reset()

        if verbose:
            print("=" * 60)
            print("UR5e 键盘遥操作启动")
            print("=" * 60)
            print("按键说明:")
            print("  W / S  : +X / -X (世界坐标系X轴)")
            print("  A / D  : +Y / -Y (世界坐标系Y轴)")
            print("  Q / E  : +Z / -Z (世界坐标系Z轴)")
            print("  空格键 : 复位到初始位姿")
            print("=" * 60)
            print(f"每次移动步长: {pos_action_scale*1000:.1f} mm")
            print(f"力控模式: {'启用' if is_use_force_control else '禁用'}")
            print(f"频率分离: 策略1次/按键 = 导纳{admittance_ratio}次 (sub_steps={self.admittance_sub_steps})")

    def _reset(self):
        """复位到机器人初始位姿"""
        mujoco.mj_resetData(self.model, self.data)
        initial_joint_pos = np.radians([193.37, -106.87, -103.69, -59.44, 90.0, 13.37])
        self.data.qpos[:6] = initial_joint_pos
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # actuator是motor（力矩控制），必须用qfrc_bias补偿重力
        for _ in range(100):
            self.data.ctrl[:6] = self.data.qfrc_bias[:6]
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # 复位控制器状态
        self.ur5e_controller.reset()

    def get_current_eef_pose(self):
        """获取当前末端位姿 (位置，四元数)"""
        current_pos = self.data.site_xpos[self.eef_site_id].copy()
        current_rot = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        current_quat = rotmat_to_quat(current_rot)
        return current_pos, current_quat

    def step_keyboard(self, dx, dy, dz):
        """根据键盘增量执行一步控制（分段插值 + 1/10导纳频率）

        1次按键 = 1个策略步，分为admittance_ratio段，每段计算一次导纳。
        总位移4mm均匀分配到10段，每段0.4mm，避免惯性力过大。
        """
        start_pos, start_quat = self.get_current_eef_pose()
        dest_pos = start_pos + np.array([dx, dy, dz]) * self.pos_action_scale
        dest_quat = start_quat  # 姿态保持不变

        # 确保目标四元数与起始四元数在同一半球
        if np.dot(start_quat, dest_quat) < 0:
            dest_quat = -dest_quat

        start_time = time.perf_counter()

        self.ur5e_controller.update(
            dest_pos, dest_quat,
            admittance_ratio=self.admittance_ratio,
            admittance_sub_steps=self.admittance_sub_steps,
        )

        delta_time = time.perf_counter() - start_time

        # 打印信息
        new_pos, _ = self.get_current_eef_pose()
        cal_f = self.ur5e_controller.calibrated_ft
        adm_dx = self.ur5e_controller.admittance_dx
        move_dist = np.linalg.norm(new_pos - start_pos) * 1000
        print(f"Command: [{dx:+.0f}, {dy:+.0f}, {dz:+.0f}]  "
              f"Target: {dest_pos.round(4)}  Actual: {new_pos.round(4)}  "
              f"move={move_dist:.1f}mm  dt:{delta_time*1000:.1f}ms")
        if self.ur5e_controller.is_use_force_control:
            print(f"  校准力: {cal_f.round(3)}  导纳偏移: {adm_dx.round(5)}")
        return True

    def key_callback(self, keycode):
        """键盘回调函数"""
        self.key_pressed = keycode

    def run(self):
        """启动主循环"""
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            while viewer.is_running():
                if self.key_pressed is not None:
                    dx, dy, dz = 0, 0, 0
                    # W/S: X轴
                    if self.key_pressed in (ord('w'), ord('W')):
                        dx = 1
                    elif self.key_pressed in (ord('s'), ord('S')):
                        dx = -1
                    # A/D: Y轴
                    elif self.key_pressed in (ord('a'), ord('A')):
                        dy = 1
                    elif self.key_pressed in (ord('d'), ord('D')):
                        dy = -1
                    # Q/E: Z轴
                    elif self.key_pressed in (ord('q'), ord('Q')):
                        dz = 1
                    elif self.key_pressed in (ord('e'), ord('E')):
                        dz = -1
                    # 空格复位
                    elif self.key_pressed == 32:
                        print("\n>>> 复位到初始位姿")
                        self._reset()
                        self.key_pressed = None
                        viewer.sync()
                        continue

                    if dx != 0 or dy != 0 or dz != 0:
                        self.step_keyboard(dx, dy, dz)
                    self.key_pressed = None
                viewer.sync()


if __name__ == "__main__":
    teleop = TeleopMujocoKeyboard(
        pos_action_scale=0.004,  # 每次按键移动4mm（10段×0.4mm，力传感器=0）
        is_use_force_control=True,
    )
    teleop.run()



