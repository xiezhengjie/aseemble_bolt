import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt
from utils.math_utils import *
from algorithm.ur5e_ik import UR5eIK
from algorithm.force_calibration import ForceCalibrationSim
import multiprocessing as mp

def _cross(a, b):
    """np.cross 的快速替代（np.cross 的 Python 封装开销 ~80µs，热路径不可接受）"""
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

def _quat_to_rotmat_fast(q):
    """[w,x,y,z] 四元数直接转旋转矩阵（替代 scipy Rotation，~57µs → ~3µs）"""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    s = 2.0 / n if n > 0.0 else 0.0
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])

def _rotmat_to_rotvec_fast(Re):
    """旋转矩阵转旋转向量（替代 scipy from_matrix().as_rotvec()，~415µs → ~10µs）。
    仅用于误差旋转（|θ| 远小于 π），θ→0 时取一阶近似。"""
    ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    cos = (Re[0, 0] + Re[1, 1] + Re[2, 2] - 1.0) * 0.5
    cos = min(1.0, max(-1.0, cos))
    angle = np.arccos(cos)
    if angle < 1e-8:
        return 0.5 * ax
    return (angle / (2.0 * np.sin(angle))) * ax

class AdmittanceController:
    """导纳控制器：m·ẍ + b·ẋ + k·x = f_net（半隐式欧拉积分），速度先行。"""

    AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz')

    def __init__(self, m, j, k_t, k_r, zeta_t, zeta_r, dt=0.001,
                 max_pos_acc=None, max_rot_acc=None,   # None → 0.9倍力限幅折算: 0.9*max_force/m, 0.9*max_torque/j
                 max_force=50.0, max_torque=5.0,
                 force_deadzone=0.0, torque_deadzone=0.0,
                 axis_mask=(1, 1, 1, 1, 1, 1),
                 b_t=None, b_r=None,
                 b0_t=500.0, b0_r=20.0):
        self.m = self._vec3(m, 'm')
        self.j = self._vec3(j, 'j')
        self.k_t = self._vec3(k_t, 'k_t')
        self.k_r = self._vec3(k_r, 'k_r')
        z_t = self._vec3(zeta_t, 'zeta_t')
        z_r = self._vec3(zeta_r, 'zeta_r')

        if b_t is None:
            self.b_t = z_t * 2 * np.sqrt(self.k_t * self.m)
            self.b_t[self.k_t == 0] = b0_t   # 零刚度轴 = 纯阻尼器，阻尼显式兜底
        else:
            self.b_t = self._vec3(b_t, 'b_t')
        if b_r is None:
            self.b_r = z_r * 2 * np.sqrt(self.k_r * self.j)
            self.b_r[self.k_r == 0] = b0_r
        else:
            self.b_r = self._vec3(b_r, 'b_r')

        # 加速度限幅默认取 0.9 倍力限幅折算值（按最大 m/j 保守取标量），
        # 使 acc 限幅稳定先于力限幅生效：acc 为主限幅，力限幅仅兜底
        if max_pos_acc is None:
            max_pos_acc = 0.9 * max_force / float(np.max(self.m))
        if max_rot_acc is None:
            max_rot_acc = 0.9 * max_torque / float(np.max(self.j))

        self.dt, self.max_pos_acc, self.max_rot_acc = dt, max_pos_acc, max_rot_acc
        self.max_force, self.max_torque = max_force, max_torque
        self.force_deadzone, self.torque_deadzone = force_deadzone, torque_deadzone
        self.mask = np.ones(6)
        self.reset()                       # 先建状态，再应用 mask
        self.set_axis_mask(axis_mask)

    @staticmethod
    def _vec3(x, name):
        a = np.asarray(x, float).ravel()
        if a.size == 1:
            return np.full(3, a.item())
        if a.size == 3:
            return a.copy()
        raise ValueError(f"{name} 应为标量或 3 维向量，得到长度 {a.size}")

    def set_axis_mask(self, mask):
        """
            运行时切换导纳轴：mask[i]=1 导纳，0 刚性。
            被关断的轴速度清零：k>0 的轴偏移平滑回零，k=0 的轴冻结当前偏移。
        """
        new = np.asarray(mask, float).ravel()
        if new.size != 6:
            raise ValueError(f"axis_mask 长度应为 6，得到 {new.size}")
        off = (self.mask > 0) & (new == 0)     # 本次被关断的轴
        self.x[6:12][off] = 0.0                # 位置不动，只杀速度 → 输出连续
        self.mask = new

    def set_axes(self, *names):
        """按轴名开启，其余关闭，如 set_axes('x','y','rx','ry')。"""
        bad = set(names) - set(self.AXES)
        if bad:
            raise ValueError(f"未知轴名 {bad}，可选 {self.AXES}")
        self.set_axis_mask([1.0 if a in names else 0.0 for a in self.AXES])

    def _apply_deadzone(self, f):
        dz = np.array([self.force_deadzone]*3 + [self.torque_deadzone]*3)
        return np.sign(f) * np.maximum(np.abs(f) - dz, 0.0)

    @staticmethod
    def _clip_acc(a, limit):
        n = np.linalg.norm(a)
        if n > limit:
            a *= limit / n

    def _derivative(self, f_ext):
        f_net = self._apply_deadzone((f_ext - self.f0) * self.mask)
        # 限制导纳可接受的最大力/力矩（保方向限幅，与 _clip_acc 同一形式）
        self._clip_acc(f_net[:3], self.max_force)
        self._clip_acc(f_net[3:], self.max_torque)
        self.dx[:3], self.dx[3:6] = self.x[6:9], 0.0
        self.dx[6:9]  = (f_net[:3] - self.k_t*self.x[:3] - self.b_t*self.x[6:9]) / self.m
        self.dx[9:12] = (f_net[3:] - self.k_r*self.x[3:6] - self.b_r*self.x[9:12]) / self.j
        self._clip_acc(self.dx[6:9], self.max_pos_acc)
        self._clip_acc(self.dx[9:12], self.max_rot_acc)

    def _integral(self, dt):
        self.x[6:12] += self.dx[6:12] * dt
        self.x[:3] += self.x[6:9] * dt

    @staticmethod
    def _gen_omega(w):
        return np.array([[0, -w[0], -w[1], -w[2]],
                         [w[0], 0, w[2], -w[1]],
                         [w[1], -w[2], 0, w[0]],
                         [w[2], w[1], -w[0], 0]])

    def _quat_update(self, dt):
        self.quat += 0.5 * self._gen_omega(self.x[9:12]) @ self.quat * dt
        self.quat /= np.linalg.norm(self.quat)
        self.x[3:6] = quat_to_euler(self.quat)

    def get_output(self, f_ext, dt=None):
        dt = self.dt if dt is None else dt
        self._derivative(f_ext)
        self._integral(dt)
        self._quat_update(dt)
        return self.x[:3].copy(), self.quat.copy()

    def reset(self):
        self.x, self.dx, self.f0 = np.zeros(12), np.zeros(12), np.zeros(6)
        self.quat = np.array([1.0, 0, 0, 0])

    @property
    def force(self):
        return self.f0

    @force.setter
    def force(self, f0):
        self.f0[:] = np.asarray(f0, float)

class IncrementalPID:
    """增量式 PID：Δu = Kp·Δe + Ki·e·dt + Kd·Δ²e/dt。"""

    def __init__(self, kp, ki, kd, dim=6, delta_limit=None):
        self.kp = MathUtils._as_vec(kp, dim)
        self.ki = MathUtils._as_vec(ki, dim)
        self.kd = MathUtils._as_vec(kd, dim)
        self.dim = dim
        self.delta_limit = None if delta_limit is None else MathUtils._as_vec(delta_limit, dim)
        self.setpoint = np.zeros(dim)
        self._prev = None          # [e(k-1), e(k-2)]，None 表示未初始化

    def __call__(self, measurement, dt):
        e = self.setpoint - np.asarray(measurement, np.float64)
        if self._prev is None:
            self._prev = np.stack([e, e])       # e(-1)=e(-2)=e(0) → Δe=Δ²e=0
        d1 = e - self._prev[0]
        d2 = e - 2.0 * self._prev[0] + self._prev[1]
        delta = self.kp * d1 + self.ki * e * dt + self.kd * d2 / dt
        if self.delta_limit is not None:
            delta = np.clip(delta, -self.delta_limit, self.delta_limit)
        self._prev[1] = self._prev[0]
        self._prev[0] = e
        return delta

    def reset(self):
        """只清历史；setpoint 保持不变，调用方按需重设。"""
        self._prev = None

class PID:
    """
        位置式 PID，支持比例/微分先行、积分限幅与输出限幅。
        P-on-measurement 首拍初始化为 -Kp·y，消除比例偏置。
    """
    def __init__(self, kp, ki, kd, output_limit=None, dim=6,
                 proportional_on_measurement=False,
                 differential_on_measurement=False,
                 int_limit=None):
        self.kp = MathUtils._as_vec(kp, dim)
        self.ki = MathUtils._as_vec(ki, dim)
        self.kd = MathUtils._as_vec(kd, dim)
        self.dim = dim
        self.output_limit = None if output_limit is None else MathUtils._as_vec(output_limit, dim)
        self.int_limit = (MathUtils._as_vec(int_limit, dim) if int_limit is not None
                          else (self.output_limit * 0.8 if self.output_limit is not None else None))
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.setpoint = np.zeros(dim)
        self._proportional = np.zeros(dim)
        self._integral = np.zeros(dim)
        self._e_prev = None
        self._input_prev = None

    def __call__(self, measurement, dt):
        y = np.asarray(measurement, np.float64)
        e = self.setpoint - y

        if self._input_prev is None:            # 首拍：微分置零
            d_input = np.zeros(self.dim)
            d_error = np.zeros(self.dim)
            if self.proportional_on_measurement:
                self._proportional = -self.kp * y
        else:
            d_input = y - self._input_prev
            d_error = e - self._e_prev

        # 比例项
        if self.proportional_on_measurement:
            self._proportional -= self.kp * d_input
        else:
            self._proportional = self.kp * e

        # 积分项(限幅)
        self._integral += self.ki * e * dt
        if self.int_limit is not None:
            self._integral = np.clip(self._integral, -self.int_limit, self.int_limit)
        derivative = (-self.kd * d_input / dt if self.differential_on_measurement
                      else self.kd * d_error / dt)
        
        # 输出项(限幅)
        output = self._proportional + self._integral + derivative
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        self._e_prev = e
        self._input_prev = y
        return output

    def reset(self):
        """只清动态状态；setpoint 保持不变，调用方按需重设。"""
        self._proportional = np.zeros(self.dim)
        self._integral = np.zeros(self.dim)
        self._e_prev = None
        self._input_prev = None

class CTCController:
    """
        计算力矩控制：τ = qfrc_bias + M(q)·qdd_ref，qdd_ref = qdd_des + Kp·e_p + Kd·e_v。
        kd=None 时取临界阻尼 kd=2√kp；每次调用推进 1 步 mj_step。
    """
    def __init__(self, model, data, kp, kd=None, tau_limit=None, arm_dof=6, damping_comp=True):
        self.model, self.data, self.arm_dof = model, data, arm_dof
        self.kp = MathUtils._as_vec(kp, arm_dof)
        self.kd = MathUtils._as_vec(2 * np.sqrt(self.kp) if kd is None else kd, arm_dof)
        self.tau_limit = None if tau_limit is None else MathUtils._as_vec(tau_limit, arm_dof)
        # 抵消 MuJoCo 被动阻尼 -d·qd；不补偿则置零
        self.dof_damping = model.dof_damping[:arm_dof].copy() if damping_comp else np.zeros(arm_dof)
        self._buf_in = np.zeros(model.nv)    # mj_mulM 要求 nv 维输入/输出
        self._buf_out = np.zeros(model.nv)
        self.last_ctrl_torque = np.zeros(arm_dof)

    def __call__(self, q_des, qd_des=None, qdd_des=None):
        q, qd = self.data.qpos[:self.arm_dof], self.data.qvel[:self.arm_dof]
        qd_des = np.zeros(self.arm_dof) if qd_des is None else qd_des
        qdd_des = np.zeros(self.arm_dof) if qdd_des is None else qdd_des

        # 参考加速度 = 前馈 + 误差反馈
        qdd_ref = qdd_des + self.kp * (np.asarray(q_des) - q) + self.kd * (np.asarray(qd_des) - qd)

        # τ = 重力/科氏偏置 + 惯量项 + 阻尼补偿
        self._buf_in[:self.arm_dof] = qdd_ref
        mj.mj_mulM(self.model, self.data, self._buf_out, self._buf_in)
        tau = self.data.qfrc_bias[:self.arm_dof] + self._buf_out[:self.arm_dof] + self.dof_damping * qd

        if self.tau_limit is not None:
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        self.last_ctrl_torque = tau
        self.data.ctrl[:self.arm_dof] = tau
        mj.mj_step(self.model, self.data)

    def reset(self):
        self.last_ctrl_torque.fill(0)

class UR5ePIDController:
    """
        单层位置 PID 直接输出力矩。
        τ = PID(q) + Kd·qd_des + qfrc_bias + M(q)·qdd_des
    """
    def __init__(self, model, data, pos_p, pos_i, pos_d, arm_dof=6,
                 ctrl_steps=1, tau_limit=None):
        self.model, self.data, self.arm_dof = model, data, arm_dof
        self.ctrl_steps = ctrl_steps
        self.tau_limit = None if tau_limit is None else MathUtils._as_vec(tau_limit, arm_dof)
        self.kd = MathUtils._as_vec(pos_d, arm_dof)
        self.pid = PID(MathUtils._as_vec(pos_p, arm_dof), MathUtils._as_vec(pos_i, arm_dof), self.kd,
                       tau_limit, differential_on_measurement=True)
        self.pid.setpoint = self.data.qpos[:].copy()
        self._buf_in = np.zeros(model.nv)      # mj_mulM 要求 nv 维
        self._buf_out = np.zeros(model.nv)
        self.last_pid_out = np.zeros(arm_dof)
        self.last_ctrl_torque = np.zeros(arm_dof)

    def _inertia_torque(self, qdd):
        self._buf_in[:self.arm_dof] = qdd
        mj.mj_mulM(self.model, self.data, self._buf_out, self._buf_in)
        return self._buf_out[:self.arm_dof]

    def __call__(self, desired_pos, desired_vel=None, desired_acc=None):
        self.pid.setpoint = np.asarray(desired_pos, float)
        dt = self.model.opt.timestep
        for _ in range(self.ctrl_steps):
            tau_pid = self.pid(self.data.qpos[:self.arm_dof], dt)
            tau = tau_pid + self.data.qfrc_bias[:self.arm_dof]
            if desired_vel is not None:
                tau = tau + self.kd * np.asarray(desired_vel, float)
            if desired_acc is not None:
                tau = tau + self._inertia_torque(np.asarray(desired_acc, float))
            if self.tau_limit is not None:
                tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            self.last_pid_out = tau_pid.copy()
            self.last_ctrl_torque = tau.copy()
            self.data.ctrl[:self.arm_dof] = tau
            mj.mj_step(self.model, self.data)

    def reset(self):
        self.pid.reset()
        self.pid.setpoint = self.data.qpos[:self.arm_dof].copy()  # 防复位后跳变
        self.last_pid_out = np.zeros(self.arm_dof)
        self.last_ctrl_torque = np.zeros(self.arm_dof)

class TrajectoryGenerator:
    """
        流式 waypoint 五次多项式轨迹生成器：每段固定 n_steps 个点，段间 C2 连续。
        end_vel='secant' 时段末速度按割线法自动估算，过 waypoint 不停车。
    """
    def __init__(self, n_dof: int, end_vel: str = 'stop', alpha: float = 0.8, v_max: float = 1.57):
        self.n_dof = n_dof
        self.end_vel = end_vel      # 'stop'：段末 qd=0；'secant'：割线法估算
        self.alpha = alpha          # 割线速度系数（0~1，越大越激进）
        self.v_max = v_max          # 段末速度钳位 (rad/s)
        self.reset()

    def reset(self):
        self.active = False
        self.c = None
        self.q_end = np.zeros(self.n_dof)
        self.qd_end = np.zeros(self.n_dof)
        self.qdd_end = np.zeros(self.n_dof)

    def replan(self, q_target, n_steps, dt,
               qd_target=None, qdd_target=None,
               q_start=None, qd_start=None, qdd_start=None):
        """
            生成新段；q_start 为 None 时自动继承前段末端状态。
            qd_target 为 None 时按 self.end_vel 模式决定段末速度。
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        self.n_steps, self.dt, self.T = n_steps, dt, n_steps * dt

        q1 = np.asarray(q_target, float)
        qdd1 = np.zeros(self.n_dof) if qdd_target is None else np.asarray(qdd_target, float)
        if q_start is None:
            if not self.active:
                raise RuntimeError("首段需提供 q_start")
            q0, qd0, qdd0 = self.q_end, self.qd_end, self.qdd_end
        else:
            q0 = np.asarray(q_start, float)
            qd0 = np.zeros(self.n_dof) if qd_start is None else np.asarray(qd_start, float)
            qdd0 = np.zeros(self.n_dof) if qdd_start is None else np.asarray(qdd_start, float)

        if qd_target is not None:                       # 显式给定优先
            qd1 = np.asarray(qd_target, float)
        elif self.end_vel == 'secant':                  # 割线法：本段平均速度 × α，钳位
            qd1 = np.clip(self.alpha * (q1 - q0) / self.T, -self.v_max, self.v_max)
        else:                                           # 'stop'：段末静止
            qd1 = np.zeros(self.n_dof)

        T = self.T
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        dq = q1 - (q0 + qd0*T + 0.5*qdd0*T2)
        dv = qd1 - (qd0 + qdd0*T)
        da = qdd1 - qdd0
        c3 = 10*dq/T3 - 4*dv/T2 + 0.5*da/T
        c4 = -15*dq/T4 + 7*dv/T3 - da/T2
        c5 = 6*dq/T5 - 3*dv/T4 + 0.5*da/T3
        self.c = np.stack([q0, qd0, 0.5*qdd0, c3, c4, c5], axis=-1)

        self.q_end, self.qd_end, self.qdd_end = self._eval(self.c, self.T)
        self.active = True
        return self

    def get_points(self):
        """返回 (q, qd, qdd, t)，采样区间 (0, T]，不含起点避免与上段重复。"""
        if not self.active:
            raise RuntimeError("先调用 replan()")
        ts = np.arange(1, self.n_steps + 1) * self.dt
        k = np.arange(6)
        P   = ts[:, None] ** k
        Pd  = k * ts[:, None] ** np.clip(k - 1, 0, None)
        Pdd = k * (k - 1) * ts[:, None] ** np.clip(k - 2, 0, None)
        return P @ self.c.T, Pd @ self.c.T, Pdd @ self.c.T, ts

    @staticmethod
    def _eval(c, t):
        t2, t3, t4, t5 = t**2, t**3, t**4, t**5
        q   = c[...,0] + c[...,1]*t + c[...,2]*t2 + c[...,3]*t3 + c[...,4]*t4 + c[...,5]*t5
        qd  = c[...,1] + 2*c[...,2]*t + 3*c[...,3]*t2 + 4*c[...,4]*t3 + 5*c[...,5]*t4
        qdd = 2*c[...,2] + 6*c[...,3]*t + 12*c[...,4]*t2 + 20*c[...,5]*t3
        return q, qd, qdd


class UR5eController:
    def __init__(self, model, data, urdf_filename,
                 pos_p, pos_d, pos_v, vel_p, vel_i,
                 m, j, k_t, k_r,zeta_r,zeta_t,b_t,b_r,
                 arm_dof=6,
                 vel_limit=3.1416/2,
                 tau_limit=np.array([150, 150, 150, 28, 28, 28]),
                 f_0=np.zeros(6),
                 is_use_force_control=False,
                 # CTC参数
                 ctc_kp=None, ctc_kd=None,
                 # 力校准参数
                 eef_body_id=None, 
                 eef_site_id=None,
                 force_sensor_site_id=None,
                 mass=1.0, gravity=np.array([0, 0, -9.8]),
                 cutoff_freq=30, force_tr=0, torque_tr=0,
                 # 导纳死区
                 force_deadzone=0.2, torque_deadzone=0.01,
                 # 导纳可接受的最大力/力矩
                 max_force=50.0, max_torque=5.0,
                 # 导纳积分步长
                 admittance_dt=None,
                 verbose=False):

        self.model = model
        self.data = data
        self.arm_dof = arm_dof
        self.is_use_force_control = is_use_force_control
        self.verbose = verbose

        # 末端body和力传感器site ID
        if eef_body_id is None:
            eef_body_id = self.model.body('tcp_link').id
        if eef_site_id is None:
            eef_site_id = self.model.site('tcp_site').id
        if force_sensor_site_id is None:
            force_sensor_site_id = self.model.site('force_torque').id
        self.eef_body_id = eef_body_id
        self.eef_site_id = eef_site_id
        self.force_sensor_site_id = force_sensor_site_id

        if admittance_dt is None:
            admittance_dt = 20 * model.opt.timestep
        self.admittance_dt = admittance_dt

        # 力校准器
        self.ft_calibration = ForceCalibrationSim(
            data=data, g=gravity, mass=mass,
            cutoff_freq=cutoff_freq, dt=admittance_dt+model.opt.timestep,
            force_tr=force_tr, torque_tr=torque_tr
        )
        # 惯性补偿开关（标准处理链第 3 层）：UR5E_FT_INERTIA_COMP=0 关闭
        self.ft_inertia_comp = os.environ.get("UR5E_FT_INERTIA_COMP", "1") != "0"
        self._acc6 = np.zeros(6)
        # 导纳控制器
        self.admittance_controller = AdmittanceController(
            m=m, j=j, k_t=k_t, k_r=k_r,
            zeta_t=zeta_t, zeta_r=zeta_r,
            b_t=b_t, b_r=b_r,
            dt=admittance_dt,
            force_deadzone=force_deadzone, torque_deadzone=torque_deadzone,
            max_force=max_force, max_torque=max_torque,
        )
        self.admittance_controller.f0 = f_0.copy()

        # IK求解器
        self.solver = UR5eIK(urdf_filename=urdf_filename, verbose=verbose)

        # 内环控制器
        self.inner_controller = CTCController(
            model, data, kp=ctc_kp, kd=ctc_kd,
            tau_limit=tau_limit, arm_dof=arm_dof
        )
        self.tau_limit = np.asarray(tau_limit, float)

        # 控制模式: osc（操作空间控制，无 IK）或 ik（IK+CTC 关节内环）
        self.ctrl_mode = os.environ.get("UR5E_CTRL_MODE", "osc").lower()

        # ---- OSC 参数（SERL opspace 风格：J^T·Λ·ẍ + 零空间姿态 PD + 重力补偿）----
        self.osc_kp_pos = np.full(3, 2500.0)    # 位置刚度
        self.osc_kp_ori = np.full(3, 2500.0)    # 姿态刚度
        self.osc_damping_ratio = 1.0            # 临界阻尼
        self.osc_null_kp = 20.0                 # 零空间姿态刚度
        self.osc_null_kd = 2 * np.sqrt(self.osc_null_kp)
        self.osc_max_pos_acc = 200.0            # m/s^2 任务空间加速度限幅（2.0 会卡死 50ms 步进）
        self.osc_max_ori_acc = 1000.0           # rad/s^2
        self.osc_lambda_damping = 1e-2          # 奇异回避阻尼
        self.osc_max_force_fb = 6.0             # N，笛卡尔反馈力限幅（15N→过压卡入超程；10N→2%%；6N 最优）
        self.osc_max_torque_fb = 1.5            # Nm
        # 零空间目标姿态（UR5e home 键位）
        key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, 'home')
        self.osc_q_home = (model.key_qpos[key_id, :arm_dof].copy() if key_id >= 0
                           else np.zeros(arm_dof))
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._M = np.zeros((model.nv, model.nv))
        self._J = np.zeros((6, arm_dof))
        self._eye6 = np.eye(6)
        self._eye_arm = np.eye(arm_dof)
        # OSC 阻尼项预计算（kd = 2ζ√kp，ζ 运行时不变）
        self.osc_kd_pos = 2 * self.osc_damping_ratio * np.sqrt(self.osc_kp_pos)
        self.osc_kd_ori = 2 * self.osc_damping_ratio * np.sqrt(self.osc_kp_ori)

        # 轨迹生成器
        self.trajectory_generator = TrajectoryGenerator(n_dof=arm_dof, end_vel='secant', alpha=0.5, v_max=vel_limit)

        # 缓存最新校准力和导纳输出，供外部诊断
        self.calibrated_ft = np.zeros(6)
        self.max_calibrated_ft  = np.zeros(6) 
        self.admittance_dx = np.zeros(3)
        self.admittance_dq = np.array([1.0, 0, 0, 0])

        self.active = False  # 控制循环是否激活

    def reset(self):
        self.active = False
        self.calibrated_ft = np.zeros(6)
        self.max_calibrated_ft  = np.zeros(6) 
        self.admittance_dx = np.zeros(3)
        self.admittance_dq = np.array([1.0, 0, 0, 0])
        
        self.trajectory_generator.reset()
        self.admittance_controller.reset()
        self.ft_calibration.reset()
        self.inner_controller.reset()

    def reset_admittance(self):
        """仅复位导纳状态，保留PID/CTC和力校准状态"""
        self.admittance_controller.reset()

    def read_calibrate_force(self, dt=None):
        """读取力传感器并校准（去重力、去惯性力、滤波、阈值限位）"""
        R_wt = self.data.site_xmat[self.force_sensor_site_id].reshape(3, 3)
        R_ws = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        R_ts = R_wt.T @ R_ws
        
        P_ws = self.data.site_xpos[self.force_sensor_site_id].copy()
        P_wt = self.data.site_xpos[self.eef_site_id].copy()
        P_wtm = self.data.xipos[self.eef_body_id].copy()
        P_stm = R_ws.T @ (P_wtm - P_ws)
        P_ts = R_wt.T @ (P_ws - P_wt)

        force = self.data.sensor('force').data.copy()
        torque = self.data.sensor('torque').data.copy()
        f_ext = np.concatenate([force, torque])
        if dt is None:
            dt = self.model.opt.timestep

        # 校准力，在工具坐标系下进行
        self.ft_calibration.update(f_ext, R_ws, R_ts, P_stm, P_ts, dt,
                                   inertial_ft=self._tool_inertial_wrench(P_stm) if self.ft_inertia_comp else None)
        self.calibrated_ft = np.concatenate([    
            self.ft_calibration.force, self.ft_calibration.torque
        ])

        # 更新控制循环中的最大接触力（按力/力矩的范数比较）
        current_norm = np.linalg.norm(self.calibrated_ft)
        max_norm = np.linalg.norm(self.max_calibrated_ft)
        if current_norm > max_norm:
            self.max_calibrated_ft = self.calibrated_ft.copy()
        
        return self.calibrated_ft.copy()
    
    def _tool_inertial_wrench(self, P_stm):
        """工具（传感器以下整体）Newton-Euler 惯性 wrench，传感器系、关于传感器原点。
        F = m·a_com；T = I_c·α + ω×(I_c·ω) + r_com×F（仿真内参数精确，补偿无损）。"""
        m, d = self.model, self.data
        bid = self.eef_body_id
        mj.mj_objectAcceleration(m, d, mj.mjtObj.mjOBJ_BODY, bid, self._acc6, 0)
        alpha_w = self._acc6[:3]
        # MuJoCo cacc 是固有加速度（含 −g，静止时读 +9.81），须加回重力得坐标加速度
        a_org_w = self._acc6[3:] + m.opt.gravity
        omega_w = d.cvel[bid][:3]                       # 世界系角速度
        r_w = d.xipos[bid] - d.xpos[bid]                # body 原点 → CoM
        a_com = a_org_w + _cross(alpha_w, r_w) + _cross(omega_w, _cross(omega_w, r_w))
        F_w = m.body_mass[bid] * a_com
        R_b = d.ximat[bid].reshape(3, 3)                # 惯性系姿态
        I_c = R_b @ np.diag(m.body_inertia[bid]) @ R_b.T
        L_dot = I_c @ alpha_w + _cross(omega_w, I_c @ omega_w)

        R_ws = d.site_xmat[self.eef_site_id].reshape(3, 3)   # 与 read_calibrate_force 同一坐标系（tcp）
        F_s = R_ws.T @ F_w
        T_s = R_ws.T @ L_dot + _cross(P_stm, F_s)
        return np.concatenate([F_s, T_s])

    def reset_max_force_tracking(self):
        """重置最大力追踪，在控制循环开始时调用"""
        self.max_force_during_control = np.zeros(6)

    def _compute_admittance_offset(self, f_calibrated, dt):
        """计算导纳偏移"""
        # 将校准力转换为世界坐标系
        R_wt = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        f = R_wt @ f_calibrated[:3]
        t = R_wt @ f_calibrated[3:]
        ft = np.concatenate([f, t])

        if not self.is_use_force_control:
            return np.zeros(3), np.array([1.0, 0, 0, 0])
        
        if not self.is_use_force_control:
            return np.zeros(3), np.array([1.0, 0, 0, 0])
        
        dx, dq = self.admittance_controller.get_output(ft, dt=dt)
        self.admittance_dx = dx.copy()

        self.admittance_dq = dq.copy()
        return dx, dq
    
    def update(self, dst_pos, dst_quat, admittance_ratio=10, admittance_sub_steps=10):
        """移动到目标位姿"""
        if self.ctrl_mode == "osc":
            return self.update_osc(dst_pos, dst_quat, admittance_ratio, admittance_sub_steps)
        dt = self.model.opt.timestep * admittance_sub_steps   # 点间隔 = 每段实际仿真时长
        ts = self.model.opt.timestep
        q_now = self.data.qpos[:self.arm_dof].copy()

        q_target = self.solver.ik(dst_pos, dst_quat, q_now)
        if q_target is None:
            q_target = q_now.copy()

        if not self.active:
            self.active = True
            self._q_des_prev = None                            # 差分链随首段重建
            self.trajectory_generator.replan(q_target, admittance_ratio, dt,
                                            q_start=q_now, qd_start=np.zeros(self.arm_dof))
        else:
            self.trajectory_generator.replan(q_target, admittance_ratio, dt)
        q_list, qd_list, qdd_list, _ = self.trajectory_generator.get_points()

        for k in range(len(q_list)):
            f = self.read_calibrate_force(dt=dt)
            dx, dq = self._compute_admittance_offset(f, dt)

            pos, quat = self.solver.fk(q_list[k])
            pos = pos + dx
            quat = quat_multiply(dq, quat)

            q_des = self.solver.ik(pos, quat, self.data.qpos[:self.arm_dof].copy())
            if q_des is None:                                  # IK 失败：保持上次参考，断差分链
                q_des = self._q_des_prev[0] if self._q_des_prev is not None else q_now
                qd_des, qdd_des = np.zeros(self.arm_dof), np.zeros(self.arm_dof)
            elif self._q_des_prev is None:                     # 首点：用生成器前馈
                qd_des, qdd_des = qd_list[k], qdd_list[k]
            else:                                              # 差分：前馈对应调整后参考
                q_prev, qd_prev = self._q_des_prev
                dq_j = (q_des - q_prev + np.pi) % (2 * np.pi) - np.pi   # 角度回绕
                qd_des = dq_j / dt
                qdd_des = (qd_des - qd_prev) / dt

            # 子步斜坡：参考从 q_des − qd·dt 匀速走到 q_des，与常值前馈一致
            for j in range(1, admittance_sub_steps + 1):
                q_ref = q_des + qd_des * ((j - admittance_sub_steps) * ts)
                self.inner_controller(q_ref, qd_des=qd_des, qdd_des=qdd_des)

            self._q_des_prev = (q_des.copy(), qd_des.copy())
            
    # ============================================================
    #  OSC 路径：操作空间控制（SERL opspace 风格），全程无 IK
    # ============================================================
    def _osc_tau(self, pos_t, quat_t, vel_ff=None):
        """计算 OSC 力矩：τ = JᵀΛẍ_des + Nᵀτ₀ + qfrc_bias；vel_ff 为目标任务空间速度前馈"""
        m, d, n = self.model, self.data, self.arm_dof

        mj.mj_jacSite(m, d, self._jacp, self._jacr, self.eef_site_id)
        J = self._J
        J[:3] = self._jacp[:, :n]
        J[3:] = self._jacr[:, :n]
        mj.mj_fullM(m, self._M, d.qM)
        M = self._M[:n, :n]

        q = d.qpos[:n]
        qd = d.qvel[:n]
        x_pos = d.site_xpos[self.eef_site_id]
        R_cur = d.site_xmat[self.eef_site_id].reshape(3, 3)

        # 任务空间位姿误差（姿态误差用旋转矢量，世界系）
        e_pos = pos_t - x_pos
        R_t = _quat_to_rotmat_fast(quat_t)
        e_ori = _rotmat_to_rotvec_fast(R_t @ R_cur.T)

        # 任务空间速度（含前馈：阻尼作用在速度误差上，消除跟踪滞后）
        dx = J @ qd
        v_des = np.zeros(6) if vel_ff is None else np.asarray(vel_ff, float)
        # 接触门控：接触时前馈阻尼项会变成持续推力（kd·v_des），必须撤除
        f = self.calibrated_ft
        if np.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]) > 2.0:
            v_des = np.zeros(6)
        ddx_pos = self.osc_kp_pos * e_pos + self.osc_kd_pos * (v_des[:3] - dx[:3])
        ddx_ori = self.osc_kp_ori * e_ori + self.osc_kd_ori * (v_des[3:] - dx[3:])
        na = np.sqrt(ddx_pos @ ddx_pos)
        nb = np.sqrt(ddx_ori @ ddx_ori)
        if na > self.osc_max_pos_acc:
            ddx_pos *= self.osc_max_pos_acc / na
        if nb > self.osc_max_ori_acc:
            ddx_ori *= self.osc_max_ori_acc / nb
        ddx_des = np.concatenate([ddx_pos, ddx_ori])

        # 动力学一致伪逆：J̄ = M⁻¹JᵀΛ，Λ = (J M⁻¹ Jᵀ + λ²I)⁻¹
        MinvJt = np.linalg.solve(M, J.T)
        Lambda = np.linalg.inv(J @ MinvJt + (self.osc_lambda_damping ** 2) * self._eye6)
        Jbar = MinvJt @ Lambda

        # 笛卡尔反馈力限幅（OSC 标准做法）：冲击/卡阻时自动软化
        F_fb = Lambda @ ddx_des
        f3 = F_fb[:3]
        t3 = F_fb[3:]
        nf = np.sqrt(f3 @ f3)
        nt = np.sqrt(t3 @ t3)
        if nf > self.osc_max_force_fb:
            F_fb[:3] *= self.osc_max_force_fb / nf
        if nt > self.osc_max_torque_fb:
            F_fb[3:] *= self.osc_max_torque_fb / nt
        tau_task = J.T @ F_fb
        # 零空间姿态稳定（拉向 home，阻尼关节速度）
        tau0 = self.osc_null_kp * (self.osc_q_home - q) - self.osc_null_kd * qd
        N = self._eye_arm - Jbar @ J
        tau = tau_task + N.T @ tau0 + d.qfrc_bias[:n]

        return np.clip(tau, -self.tau_limit, self.tau_limit)

    @staticmethod
    def _quat_slerp(q0, q1, s):
        """两四元数球面插值（替代 scipy Slerp，热路径每次策略步调用 10 次）"""
        dot = q0 @ q1
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:                     # 近重合：线性插值 + 归一化
            r = q0 + s * (q1 - q0)
            return r / np.sqrt(r @ r)
        th = np.arccos(min(1.0, dot))
        s_th = np.sin(th)
        return (np.sin((1.0 - s) * th) / s_th) * q0 + (np.sin(s * th) / s_th) * q1

    def update_osc(self, dst_pos, dst_quat, admittance_ratio=10, admittance_sub_steps=10):
        """OSC 移动到目标位姿：笛卡尔斜坡 + 导纳偏移 + 每 1ms 力矩控制"""
        dt = self.model.opt.timestep * admittance_sub_steps

        start_pos = self.data.site_xpos[self.eef_site_id].copy()
        start_quat = rotmat_to_quat(self.data.site_xmat[self.eef_site_id].reshape(3, 3))
        T_total = admittance_ratio * dt
        vel_ff = np.zeros(6)
        vel_ff[:3] = (dst_pos - start_pos) / T_total               # 斜坡线速度前馈

        for k in range(1, admittance_ratio + 1):
            f = self.read_calibrate_force(dt=dt)
            dx, dq = self._compute_admittance_offset(f, dt)

            s = k / admittance_ratio
            pos_t = start_pos + (dst_pos - start_pos) * s + dx
            quat_ramp = self._quat_slerp(start_quat, dst_quat, s)
            quat_t = quat_multiply(dq, quat_ramp)

            for _ in range(admittance_sub_steps):
                self.data.ctrl[:self.arm_dof] = self._osc_tau(pos_t, quat_t, vel_ff=vel_ff)
                mj.mj_step(self.model, self.data)
