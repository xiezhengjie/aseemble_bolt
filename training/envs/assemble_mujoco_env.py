"""机械臂装配任务的 MuJoCo 环境。

基于 Gymnasium 标准接口，整合力校准→导纳控制→CTC内环的完整力控流程。

控制流程 (频率分离):
    策略层(20Hz): 策略输出动作(7维) → 末端目标位姿 (3mm/步)
    导纳层(200Hz): 1个策略步分为10段，每段移动0.3mm
        每段: 计算导纳偏移(dt=子轨迹时间=5ms) → 调整目标 → min-jerk短轨迹 → CTC跟踪
    仿真层(1000Hz): CTC内环每仿真步执行

观测 (10维, 环境内已物理归一化):
    [dx, dy, dz, yaw, fx, fy, fz, tx, ty, tz] / scale
    scale = [W_x, W_y, W_z, YAW_MAX, F_max×3, T_max×3]
    - 前3维: diag(W)⁻¹(p−p_g)，W 取 workspace 各轴半幅
    - 第4维: Δyaw / YAW_MAX
    - 后6维: diag(F_max)⁻¹F，F_max 用力/力矩终止阈值
    原始相对量仍可由 ``_compute_raw_obs()`` 取得（录制旧数据兼容 / 诊断）。

环境始终返回单帧 10 维。训练时用 gymnasium.wrappers.FrameStackObservation
叠成 (T, 10)，供 GRU 策略按时间步读取；录制演示不要叠帧。

动作空间 (6维, [-1, 1]):
    - 前3维: 末端位置增量 (× pos_action_scale=0.004, 即4mm/步)
    - 后6维: 姿态增量欧拉角 [roll, pitch, yaw] (× ori_action_scale)
"""
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目根目录：本文件位于 training/envs/，向上两级即项目根
# （不依赖 cwd；也不引入 rl_utils，避免把 torch 拖进 env 模块）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import mujoco
import mujoco.viewer
import copy
from gymnasium import Env
from gymnasium.spaces import Box

from algorithm.controllers import UR5eController
from utils.math_utils import (
    rotmat_to_quat, quat_multiply, euler_to_quat,
    rotmat_to_euler, normalize_euler,
)


class AssembleMuJoCoEnv(Env):
    """UR5e 装配任务 MuJoCo 环境。

    使用 min-jerk 插值生成轨迹，CTC 作为内环控制器，
    可选导纳控制实现外力顺应。力校准、导纳、CTC 均封装在 UR5eController 中。
    """
    DEFAULT_XML_PATH = str(PROJECT_ROOT / "mjcf" / "ur5e_assemble_sence.xml")
    DEFAULT_URDF_PATH = str(PROJECT_ROOT / "urdf" / "ur5e_assemble.urdf")
    # 预设初始位姿: (关节角[度], 末端位置[m], 末端欧拉角[度])
    DEFAULT_INITIAL_POSE = (
        [193.37, -106.87, -103.69, -59.44, 90.0, 13.37],
        [0.5764, 0, 0.16818],
        [-180, 0, 0], 
    )
    HOLE_DEPTH = 0.008
    HOLE_WIDTH = 0.008
    YAW_MAX = np.radians(30.0)
    def __init__(
        self,
        # --- 环境配置 ---
        xml_path=DEFAULT_XML_PATH,
        urdf_path=DEFAULT_URDF_PATH,
        render_mode=None,
        max_episodic_steps=400,

        # --- 目标与初始位姿 ---
        target_offset=np.array([-0.08247, 0, 0.02218, -85]),
        target_range = np.array([[-0.001, 0.001],     # x 轴范围
                                [-0.001, 0.001],      # y 轴范围
                                [-0.003, 0.003],      # z 轴范围
                                [-5.0, 5.0],          # yaw 范围
                    ]),
        initial_offset=np.array([0.0, 0.0, 0.1]),   
        initial_range=np.array([[-0.001, 0.001],               # x 轴范围
                                [-0.001, 0.001],               # y 轴范围
                                [-0.001, 0.001],               # z 轴范围
                                [-10.0, 10.0],            # rx 范围
                                [-10.0, 10.0],            # ry 范围    
                                [-5.0, 5.0]]),             # rz 范围
        initial_eef_pose=DEFAULT_INITIAL_POSE,

        # --- 任务判定阈值 ---
        assembly_depth_threshold=0.0045,
        success_pos_threshold=0.004,
        success_reach_hole_threshold=0.01044,
        success_search_hole_threshold=1,
        success_force_threshold = 1.5,
        success_torque_threshold = 0.05,
        force_threshold_terminate=50.0,
        torque_threshold_terminate=5.0,

        # --- 工作空间 ---
        workspace=np.array([[-0.04, -0.04, -0.009],
                            [0.04, 0.04, 0.16]]),
        # workspace_max=np.array([0.04, 0.04, 0.1118]),
        # workspace_max=np.array([0.11, 0.11, 0.21]),

        # --- 奖励权重 ---
        weight_force_torque=0.1,
        weight_pose_error=1.0,
        weight_success=10.0,
        base_reward=10.0,
        # 全程走完 [seat_depth_lo, seat_depth_hi] 累积 +seat_depth_scale；
        seat_force_depth=0.006,
        seat_force_scale=8.0,
        seat_lateral_coef=0.05,

        # --- 动作缩放 ---
        pos_action_scale=0.0015,  # 每次最大移动1.5mm（均匀分配到10次导纳循环，每次0.15mm）
        ori_action_scale=0.01, # 每次最大旋转0.01 rad， 即1.8度

        # --- 力控开关 ---
        is_use_force_control=True,

        # --- 导纳参数（外环）---
        admittance_m=6.0,
        admittance_j=0.6,
        admittance_k_t=np.array([1800, 1800, 3000]),
        admittance_k_r=np.array([12, 6, 20]),
        admittance_zeta_t=2.2,
        admittance_zeta_r=1.2,
        admittance_b_t=None,
        admittance_b_r=None,
        admittance_force_deadzone=0.2,
        admittance_torque_deadzone=0.01,
        f_0=np.zeros(6),

        # --- 力校准参数 ---
        cutoff_freq=30,
        force_threshold_sensor=0,
        torque_threshold_sensor=0,

        # --- 轨迹参数 ---
        traj_max_vel=3.14,
        traj_settle_steps=2,  # 预留2个静止点，实际轨迹步数 = max_steps - settle_steps - 2

        # --- 控制周期 ---
        force_ctrl_steps=50,   # 每个策略步执行的仿真步数（50ms@1ms）
        admittance_ratio=10,  # 每个策略步执行的导纳循环数（10次）
    ):
        super().__init__()

        # ===== 基础配置 =====
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.max_episodic_steps = max_episodic_steps
        self.is_counting = True
        self.current_step = 0
        self.current_state = 0  # 0接近(R) 1搜索(S) 2插入(I) 3成功(T) 4失败(F)
        self.prev_state = 0
        self.is_teleoperation = False

        # ===== 末端与传感器ID =====
        self.eef_body_id = self.model.body('tcp_link').id
        self.eef_site_id = self.model.site('tcp_site').id
        self.force_sensor_id = self.model.site('force_torque').id

        # ===== 空间定义 =====
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.raw_obs_dim = 10  # [pos(3), yaw(1), force(3), torque(3)]
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.raw_obs_dim,),
            dtype=np.float32,
        )
        # 观测归一化尺度：diag(W)⁻¹(p−p_g)、diag(F_max)⁻¹F（采集即归一化）
        ws = np.asarray(workspace, dtype=np.float64)
        self.obs_W = np.array([
            max(abs(float(ws[0, 0])), abs(float(ws[1, 0]))),
            max(abs(float(ws[0, 1])), abs(float(ws[1, 1]))),
            max(abs(float(ws[0, 2])), abs(float(ws[1, 2]))),
            float(self.YAW_MAX),
        ], dtype=np.float64)
        self.obs_F_max = np.array(
            [float(force_threshold_terminate)] * 3
            + [float(torque_threshold_terminate)] * 3,
            dtype=np.float64,
        )
        self._obs_scale = np.concatenate([self.obs_W, self.obs_F_max]).astype(np.float32)

        # ===== 目标与初始位姿 =====
        self.target_offset = target_offset
        self.target_range = target_range
        self.initial_offset = initial_offset
        self.initial_range = initial_range
        joint_pos_deg, eef_pos, euler_deg = initial_eef_pose
        self.initial_joint_pos = np.radians(joint_pos_deg)
        self.initial_eef_pos = np.array(eef_pos)
        self.initial_euler = normalize_euler(np.radians(euler_deg))

        # ===== 任务阈值 =====
        self.assembly_depth_threshold = assembly_depth_threshold
        self.success_pos_threshold = success_pos_threshold
        self.success_reach_hole_threshold = success_reach_hole_threshold
        self.success_search_hole_threshold = success_search_hole_threshold
        self.success_force_threshold = success_force_threshold
        self.success_torque_threshold = success_torque_threshold
        self.force_threshold_terminate = force_threshold_terminate
        self.torque_threshold_terminate = torque_threshold_terminate

        # ===== 工作空间 =====
        self.workspace = workspace

        # ===== 奖励权重 =====
        self.weight_force_torque = weight_force_torque
        self.weight_pose_error = weight_pose_error
        self.weight_success = weight_success
        self.base_reward = base_reward
        self.seat_force_depth = float(seat_force_depth)
        self.seat_force_scale = float(seat_force_scale)
        self.seat_lateral_coef = float(seat_lateral_coef)
        self.seat_lateral_max = 0.6
        self._prev_fz_abs = 0.0
        self.fail_contact = False
        self.fail_workspace = False
        self.last_reward_terms = {}

        # ===== 动作缩放 =====
        self.pos_action_scale = pos_action_scale
        self.ori_action_scale = ori_action_scale

        self.last_pos = None
        self.last_quat = None

        # ===== 力控配置 =====
        self.is_use_force_control = is_use_force_control
        self.force_ctrl_steps = force_ctrl_steps
        self.admittance_ratio = admittance_ratio
        self.admittance_sub_steps = max(2, force_ctrl_steps // admittance_ratio) # 每次导纳计算的子轨迹步数（向下取整，至少2步）

        # ===== 轨迹参数 =====
        self.traj_max_vel = traj_max_vel
        self.traj_settle_steps = traj_settle_steps

        # ===== 控制器（力校准→导纳→OSC）=====
        # 力校准质量 = 末端工具重力对应的质量
        tool_mass = self.model.body_mass[self.eef_body_id]
        self.ur5e_controller = UR5eController(
            model=self.model,
            data=self.data,
            urdf_filename=urdf_path,
            # 导纳参数
            m=admittance_m, j=admittance_j,
            k_t=admittance_k_t, k_r=admittance_k_r,
            zeta_r=admittance_zeta_r, zeta_t=admittance_zeta_t,
            b_t=admittance_b_t, b_r=admittance_b_r,
            f_0=f_0,
            is_use_force_control=is_use_force_control,
            # 力校准
            mass=tool_mass,
            cutoff_freq=cutoff_freq,
            force_tr=force_threshold_sensor,
            torque_tr=torque_threshold_sensor,
            # 导纳死区
            force_deadzone=admittance_force_deadzone,
            torque_deadzone=admittance_torque_deadzone,
            # 导纳积分步长：导纳每仿真步调用一次
            admittance_dt=self.admittance_sub_steps * self.model.opt.timestep,
            verbose=False,
        )

        # ===== 计算目标位置 =====
        mujoco.mj_forward(self.model, self.data)
        self.catenary_pos = self.data.xpos[self.model.body('catenary').id].copy()
        self.base_target_pos = self.catenary_pos + target_offset[:3]
        self.base_target_rz  = np.radians(target_offset[3])
        self.target_pos = self.base_target_pos
        self.target_rz  = self.base_target_rz

    def _compute_raw_obs(self):
        """原始10维单帧: [相对位置(3), 偏航角(1), 校准力(6)]，不写入历史。

        相对位姿一律相对视觉随机目标 (target_pos / target_rz)，不含真实孔位。
        """
        eef_pos = self.data.site_xpos[self.eef_site_id].copy()
        relative_pos = eef_pos - self.target_pos
        relative_rz = self._get_yaw_error(use_true=False)
        ft = self.ur5e_controller.calibrated_ft.copy()
        return np.concatenate([relative_pos, [relative_rz], ft]).astype(np.float32)

    # ============================================================
    #  Gymnasium 标准接口
    # ============================================================
    def reset(self, seed=None, options=None):
        """复位环境到初始状态（对目标位姿和初始末端位姿加入随机偏移）"""
        super().reset(seed=seed)
        random_delta = options.get('random_delta', None) if options else None
        if random_delta is not None:
            random_delta = random_delta.copy()  
            random_delta[3] = np.radians(random_delta[3])
            random_delta[7:] = np.radians(random_delta[7:])
        self.current_step  = 0
        self.current_state = 0
        self.prev_state = 0
        self.seat_count = 0
        self.backwards_count = 0
        self._prev_fz_abs = 0.0
        self.fail_contact = False
        self.fail_workspace = False
        self.last_reward_terms = {}

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.initial_joint_pos
        self.data.ctrl[:6] = self.initial_joint_pos
        mujoco.mj_forward(self.model, self.data)

        # 随机化位置变化量
        if random_delta is None:
            delta1 = self.np_random.uniform(low=self.target_range[:, 0],  
                                           high=self.target_range[:, 1], 
                                           size=4)  
            delta2 = self.np_random.uniform(low=self.initial_range[:, 0],  
                                            high=self.initial_range[:, 1], 
                                            size=6)
            delta1[3]  = np.radians(delta1[3]) 
            delta2[3:] = np.radians(delta2[3:])
        else:
            delta1 = random_delta[:4]
            delta2 = random_delta[4:]
        # 随机初始目标位姿
        self.target_pos = self.base_target_pos + delta1[:3]
        self.target_rz  = self.base_target_rz + delta1[3]
        # 随机化初始末端位姿
        initial_pos = self.target_pos + self.initial_offset + delta2[:3]
        initial_euler = self.initial_euler + delta2[3:]
        initial_euler[2] += self.target_rz
        initial_euler = normalize_euler(initial_euler)
        initial_quat  = euler_to_quat(initial_euler)
        initial_quat /= np.linalg.norm(initial_quat)

        # 求解初始关节位置
        q_0 = self.data.qpos[:6].copy()
        q_target = self.ur5e_controller.solver.ik(initial_pos, initial_quat, q_0)
        self.data.qpos[:6] = q_target
        
        # 多步仿真至稳定
        for _ in range(100): 
            self.data.ctrl[:6] = self.data.qfrc_bias[:6]
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:6] = 0.0 # 强制速度归零，消除稳定过程残余速度

        # 复位控制器（导纳、力校准、内环）
        self.ur5e_controller.reset()

        # 记录参考位姿（用于mujoco_step保持位姿）
        self.ref_pos = self.data.site_xpos[self.eef_site_id].copy()
        self.ref_quat = rotmat_to_quat(
            self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        )
        # 与 _state_trans 一致，用真实孔位，避免视觉 z 偏移污染 Δdepth
        self.prev_depth = self._get_assemble_depth(use_true=True)
        self.prev_pos_err_xy = self._get_position_error_xy(use_true=True)
        self.prev_yaw_err = self._get_yaw_error(use_true=True)
        self.last_pos = self.ref_pos.copy()
        self.last_quat = self.ref_quat.copy()

        obs = self._get_observation()
        info = {
            'initial_joint_pos': self.initial_joint_pos.copy(),
            'initial_eef_pos': self.data.site_xpos[self.eef_site_id].copy(),
            'initial_euler': self.initial_euler.copy(),
            'target_pos': self.target_pos.copy(),
            'target_rz': np.degrees(self.target_rz),
        }

        # 重置环境后渲染环境
        if self.render_mode == 'human':
            self.render()

        return obs, info

    def step(self, action):
        """执行一步策略动作"""
        self.current_step += 1

        # 动作转目标位姿
        dst_pos, dst_quat = self._action_to_pose(action)

        # 执行控制（1个策略步 = admittance_ratio次导纳计算）
        self._execute_control(dst_pos, dst_quat)

        # 状态转移
        self.prev_state = self.current_state
        self._state_trans()

        # 计算观测、奖励、终止条件
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = self._get_info()

        # 更新上一拍状态
        ft = self.ur5e_controller.calibrated_ft
        self._prev_fz_abs = float(np.abs(ft[2]))
        self.prev_depth = self._get_assemble_depth(use_true=True)
        self.prev_pos_err_xy = self._get_position_error_xy(use_true=True)
        self.prev_yaw_err = self._get_yaw_error(use_true=True)

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info
    

    def mujoco_step(self):
        """仅推进仿真（保持当前参考位姿），不执行策略动作"""
        if self.is_counting: self.current_step += 1
        self._execute_control(self.ref_pos, self.ref_quat)
        obs = self._get_observation()
        info = self._get_info()
        if self.render_mode == 'human':
            self.render()
        return obs, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.lookat[:] = self.target_pos.tolist()
            self.viewer.cam.distance = 2.5
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

     # ============================================================
    #  控制循环
    # ============================================================
    def _execute_control(self, dst_pos, dst_quat):
        """执行一个策略步的控制循环, 包含admittance_ratio次导纳计算"""
        # 重置最大力追踪
        self.ur5e_controller.reset_max_force_tracking()
        # 获取起始位姿
        start_pos = self.data.site_xpos[self.eef_site_id].copy()
        start_rotmat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        start_quat = rotmat_to_quat(start_rotmat)

        # 无位姿变化时直接用起始值
        interp_pos = dst_pos.copy()
        interp_quat = dst_quat.copy()
        if np.allclose(start_pos, dst_pos, atol=1e-6):
            interp_pos = start_pos.copy()
        if np.allclose(start_quat, dst_quat, atol=1e-6):
            interp_quat = start_quat.copy()
        
        self.ur5e_controller.update(
            interp_pos, interp_quat,
            admittance_ratio = self.admittance_ratio,
            admittance_sub_steps=self.admittance_sub_steps
        )

        # 获取最终稳态的导纳力
        self.ur5e_controller.read_calibrate_force()

    def _get_observation(self):
        """单帧 10 维观测（已物理归一化）。多帧堆叠由 FrameStackObservation 完成。"""
        raw = self._compute_raw_obs()
        return (raw / self._obs_scale).astype(np.float32)

    def _get_info(self):
        """返回诊断信息"""
        # 计算工具z轴(销钉轴)相对世界竖直方向的偏转角，用于分析工具是否垂直
        eef_rot = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        z_axis = eef_rot[:, 2] * np.sign(eef_rot[2, 2])
        angle_z = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        rz = np.degrees(np.arctan2(eef_rot[1, 0], eef_rot[0, 0]))
        return {
            'force': self.ur5e_controller.calibrated_ft[:3].copy(),
            'torque': self.ur5e_controller.calibrated_ft[3:].copy(),
            'position_error': self._get_position_error(use_true=True),
            'position_error_xy': self._get_position_error_xy(use_true=True),
            'admittance_dx': self.ur5e_controller.admittance_dx.copy(),
            'yaw_error': np.degrees(self._get_yaw_error(use_true=True)),
            'depth': self._get_assemble_depth(use_true=True),
            'angle_z': angle_z,
            'rz': rz,
            'state': self.current_state,
            'prev_state': self.prev_state,
            'success': self.current_state == 3,
            **self.last_reward_terms,
        }

    # ============================================================
    #  动作转换
    # ============================================================
    def _action_to_pose(self, action):
        """策略动作(6维): 位置增量(3) + 欧拉角增量(3)"""
        # 遥操时零动作保持上一帧目标位姿，避免 viewer 鼠标拖动导致跟随
        if self.last_pos is not None and np.allclose(action, np.zeros(6), rtol=0, atol=1e-5):
            return self.last_pos.copy(), self.last_quat.copy()

        # 获取当前位姿
        current_pos = self.data.site_xpos[self.eef_site_id].copy()
        current_quat = rotmat_to_quat(self.data.site_xmat[self.eef_site_id].reshape(3, 3))

        # 位置
        dest_pos = current_pos + action[:3] * self.pos_action_scale

        # 姿态
        dq = euler_to_quat(action[3:] * self.ori_action_scale)
        dest_quat = quat_multiply(dq, current_quat)
        if np.dot(dest_quat, current_quat) < 0:   # 与上一帧同半球，增量路径最短
            dest_quat = -dest_quat

        # 更新缓存
        self.last_pos = dest_pos.copy()
        self.last_quat = dest_quat.copy()
        return dest_pos, dest_quat

    # ============================================================
    #  误差计算
    # ============================================================
    def _target_ref(self, use_true=False):
        """目标参考：真实孔位 或 视觉随机目标。"""
        if use_true:
            return self.base_target_pos, self.base_target_rz
        return self.target_pos, self.target_rz

    def _get_position_error(self, use_true=False):
        """末端到目标的三维位置误差（米）。

        use_true=True: 相对真实孔位；False: 相对视觉随机目标。
        """
        target_pos, _ = self._target_ref(use_true)
        eef_pos = self.data.site_xpos[self.eef_site_id]
        return np.linalg.norm(eef_pos - target_pos)

    def _get_position_error_xy(self, use_true=False):
        """末端到目标的XY平面位置误差（米）。

        use_true=True: 相对真实孔位；False: 相对视觉随机目标。
        """
        target_pos, _ = self._target_ref(use_true)
        eef_pos = self.data.site_xpos[self.eef_site_id]
        return np.linalg.norm(eef_pos[:2] - target_pos[:2])

    def _get_yaw_error(self, use_true=False):
        """末端到目标的z轴角度误差（弧度）。

        use_true=True: 相对真实孔偏航；False: 相对视觉随机偏航。
        利用正六边形60°旋转对称性，将误差wrap到[-30°, 30°]。
        例如：50° → -10°（因为 50° - 60° = -10°）。
        """
        _, target_rz = self._target_ref(use_true)
        eef_rot = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        eef_rz = np.arctan2(eef_rot[1, 0], eef_rot[0, 0])
        
        err = eef_rz - target_rz
        
        # 第一步：消除多圈旋转，先wrap到[-pi, pi]
        err = np.arctan2(np.sin(err), np.cos(err))
        
        # 第二步：利用六边形60°对称，wrap到[-30°, 30°]
        period = np.deg2rad(60.0)
        k = np.round(err / period)
        err = err - k * period
        
        return float(err)

    def _get_assemble_depth(self, use_true=False):
        """装配深度 = 目标Z - 末端Z（米）。

        use_true=True: 相对真实孔口高度；False: 相对视觉随机目标高度。
        """
        target_pos, _ = self._target_ref(use_true)
        eef_pos = self.data.site_xpos[self.eef_site_id]
        return target_pos[2] - eef_pos[2]

    # ============================================================
    #  奖励：逐步距离（不再 /T）+ 成功 / 仅越界失败
    # ============================================================
    def _compute_reward(self):
        """稀疏 +80 成功；−20 仅越界。距离 clip 到 [0,1] 后逐步给，不再除以 horizon。

        力/力矩超限仍终止，但不加失败惩罚，避免把接触当主负信号。
        """
        yaw_err = self._get_yaw_error(use_true=True)
        pos_err_xy = self._get_position_error_xy(use_true=True)
        depth = self._get_assemble_depth(use_true=True)
        xy_ref = max(abs(float(self.workspace[0][0])), abs(float(self.workspace[1][0])), 1e-6)
        yaw_ref = np.deg2rad(10.0)
        depth_ref = 0.12
        d_star = float(self.HOLE_DEPTH)

        e_xy = float(np.clip(float(pos_err_xy) / xy_ref, 0.0, 1.0))
        e_yaw = float(np.clip(abs(float(yaw_err)) / yaw_ref, 0.0, 1.0))
        if float(pos_err_xy) <= 0.008:
            e_d = float(np.clip(abs(float(depth) - d_star) / depth_ref, 0.0, 1.0))
        else:
            e_d = 0.0
        r_xy = -e_xy
        r_yaw = -e_yaw
        r_depth = -e_d

        success_bonus = 80.0 if self.current_state == 3 else 0.0
        fail_penalty = -20.0 if self.fail_workspace else 0.0

        self.last_reward_terms = {
            'r_xy': r_xy,
            'r_yaw': r_yaw,
            'r_depth': r_depth,
            'r_success': float(success_bonus),
            'r_fail': float(fail_penalty),
            'fail_contact': float(self.fail_contact),
            'fail_workspace': float(self.fail_workspace),
        }
        return float(r_xy + r_yaw + r_depth + success_bonus + fail_penalty)
    

    def _state_trans(self):
        """状态机转移函数, 0: 接近(R) 1: 搜索(S) 2: 插入(I) 3: 成功(T) 4: 失败(F) """
        eef_pos = self.data.site_xpos[self.eef_site_id].copy() - self.base_target_pos
        depth = self._get_assemble_depth(use_true=True)
        pos_error = self._get_position_error(use_true=True)
        yaw_error = self._get_yaw_error(use_true=True)
        pos_error_xy = self._get_position_error_xy(use_true=True)

        ft = self.ur5e_controller.calibrated_ft.copy()
        f_norm = np.linalg.norm(ft[:3])
        t_norm = np.linalg.norm(ft[3:])
        ft_norm = np.linalg.norm([f_norm, t_norm])

        ft_max = self.ur5e_controller.max_calibrated_ft.copy()
        f_max_norm = np.linalg.norm(ft_max[:3])
        t_max_norm = np.linalg.norm(ft_max[3:]) 

        # 4: 失败(F)。力超限仍终止，但不走越界惩罚。
        contact_fail = (
            f_max_norm > self.force_threshold_terminate
            or t_max_norm > self.torque_threshold_terminate
        )
        workspace_fail = (
            np.any(eef_pos < self.workspace[0])
            or np.any(eef_pos > self.workspace[1])
        )
        if contact_fail or workspace_fail:
            if contact_fail:
                print(f"============================力超出阈值: {f_max_norm:.2f} N, 力矩超出阈值: {t_max_norm:.2f} Nm")
            if workspace_fail:
                print(f"============================超出工作空间 {eef_pos}")
            self.fail_contact = bool(contact_fail)
            self.fail_workspace = bool(workspace_fail)
            self.current_state = 4
            return

        # print(f"====================pos_error_xy:{pos_error_xy} depth:{depth} yaw误差: {np.rad2deg(np.abs(yaw_error)):.2f}°")
        # 0: 接近(R)
        if self.current_state == 0:
            if (pos_error_xy <= 0.008 and
                depth >= -0.004):
                self.current_state = 1

        # 1: 搜索(S)
        if self.current_state == 1:
            if (pos_error_xy <= 0.003 and 
                np.abs(yaw_error) <= np.deg2rad(6) and
                depth >= 0.001):
                self.current_state = 2
            elif (pos_error_xy > 0.008 or depth < -0.004):
                self.current_state = 0
        
        # 2: 插入(I)
        if self.current_state == 2:
            # # 3: 成功(T)：坐底判据：深度达标 + 深度停滞 + 法向力持续（三条件防抖）
            # if (depth >= self.assembly_depth_threshold and
            #     abs(depth - self.prev_depth) < 2e-4 and                         # 还在发向下指令但深度不再涨 = 顶到底了
            #     pos_error_xy <= 0.0015 and                  # xy 轴误差在期望范围内
            #     np.abs(ft[2]) >= self.success_force_threshold and               # 降到 success_force_threshold N 并靠持续拍数保证
            #     np.linalg.norm(ft[:2]) <= 0.3*self.success_force_threshold and  # 0.45N
            #     np.linalg.norm(ft[3:]) <= self.success_torque_threshold):
            #     self.seat_count = self.seat_count + 1 
            # else:
            #     self.seat_count = 0
            # # 连续 3 拍坐底说明成功插入，置位
            # if self.seat_count >= 3:    
            #     self.current_state = 3
            if (depth >= 0.008 and                                     
                np.abs(ft[2]) >= self.success_force_threshold and              
                np.linalg.norm(ft[:2]) <= 0.3*self.success_force_threshold ):
                self.current_state = 3
            elif (pos_error_xy > 0.003 or np.abs(yaw_error) > np.deg2rad(6) or depth < 0.001):
                self.current_state = 1


    # ============================================================
    #  终止条件
    # ============================================================
    def _check_terminated(self):
        """检查是否终止（成功或失败）"""
        if self.current_state == 3 or self.current_state == 4:
            return True
        return False

    def _check_truncated(self):
        """检查是否截断（超时）"""
        return self.current_step >= self.max_episodic_steps
