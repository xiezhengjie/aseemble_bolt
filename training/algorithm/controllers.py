import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco as mj
from utils.math_utils import *
from algorithm.ur5e_ik import UR5eIK
from algorithm.force_calibration import ForceCalibrationSim

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


class UR5eController:
    def __init__(self, model, data, urdf_filename,
                 m, j, k_t, k_r,zeta_r,zeta_t,b_t,b_r,
                 arm_dof=6,
                 tau_limit=np.array([150, 150, 150, 28, 28, 28]),
                 f_0=np.zeros(6),
                 is_use_force_control=False,
                 # 力校准参数
                 eef_body_id=None, 
                 eef_site_id=None,
                 force_sensor_site_id=None,
                 mass=1.0, gravity=np.array([0, 0, -9.81]),
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

        # IK求解器（仅用于 reset 求初始关节角；运行时控制为纯 OSC，不经过 IK）
        self.solver = UR5eIK(urdf_filename=urdf_filename, verbose=verbose)
        self.tau_limit = np.asarray(tau_limit, float)

        # ---- OSC 参数（SERL opspace 风格：J^T·Λ·ẍ + 零空间姿态 PD + 重力补偿）----
        # 轨迹速度由前馈承担，PD 只修正跟踪误差。姿态环不能过软：接触下
        # 完整 Lambda 的平移/旋转交叉项会生成力矩，过低的姿态增益会让该项压过导纳方向。
        self.osc_kp_pos = np.full(3, 5000.0)
        self.osc_kp_ori = np.full(3, 2500.0)
        self.osc_ori_ff_scale = 0.95          # 抑制高带宽姿态环的小幅速度前馈超调
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

        # 缓存最新校准力和导纳输出，供外部诊断
        self.calibrated_ft = np.zeros(6)
        self.max_calibrated_ft  = np.zeros(6) 
        self.admittance_dx = np.zeros(3)
        self.admittance_dq = np.array([1.0, 0, 0, 0])

    def reset(self):
        self.calibrated_ft = np.zeros(6)
        self.max_calibrated_ft  = np.zeros(6)
        self.admittance_dx = np.zeros(3)
        self.admittance_dq = np.array([1.0, 0, 0, 0])

        self.admittance_controller.reset()
        self.ft_calibration.reset()

    def reset_admittance(self):
        """仅复位导纳状态，保留力校准状态"""
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
        omega_w = d.cvel[bid][:3]                       # 世界系角速度
        v_com_w = d.cvel[bid][3:]                       # 世界系质心线速度
        # cacc 是质心系空间加速度，不是质心经典加速度；加回重力并补 ω×v_com。
        # 旧实现再做一次 α×r + ω×(ω×r) 搬运算会重复计入该修正。
        a_com = self._acc6[3:] + m.opt.gravity + _cross(omega_w, v_com_w)
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

        dx, dq = self.admittance_controller.get_output(ft, dt=dt)
        self.admittance_dx = dx.copy()

        self.admittance_dq = dq.copy()
        return dx, dq
    
    # ============================================================
    #  OSC：操作空间控制（SERL opspace 风格），全程无 IK
    # ============================================================
    def _osc_tau(self, pos_t, quat_t, vel_ff=None):
        """计算 OSC 力矩：τ = JᵀΛẍ_des + Nᵀτ₀ + qfrc_bias + dof_damping·qd；vel_ff 为目标任务空间速度前馈"""
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
        v_des = np.zeros(6) if vel_ff is None else np.asarray(vel_ff, float).copy()
        # 接触时线速度前馈会变成持续推力，只撤掉平移部分；旋转轨迹仍需角速度前馈。
        f = self.calibrated_ft
        if np.sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]) > 2.0:
            v_des[:3] = 0.0
        ddx_pos = self.osc_kp_pos * e_pos + self.osc_kd_pos * (v_des[:3] - dx[:3])
        ddx_ori = self.osc_kp_ori * e_ori + self.osc_kd_ori * (v_des[3:] - dx[3:])
        na = np.sqrt(ddx_pos @ ddx_pos)
        nb = np.sqrt(ddx_ori @ ddx_ori)
        if na > self.osc_max_pos_acc:
            ddx_pos *= self.osc_max_pos_acc / na
        if nb > self.osc_max_ori_acc:
            ddx_ori *= self.osc_max_ori_acc / nb
        # 动力学一致伪逆：J̄ = M⁻¹JᵀΛ，Λ = (J M⁻¹ Jᵀ + λ²I)⁻¹
        MinvJt = np.linalg.solve(M, J.T)
        Lambda = np.linalg.inv(J @ MinvJt + (self.osc_lambda_damping ** 2) * self._eye6)
        Jbar = MinvJt @ Lambda

        # 完整 Λ 中的交叉项负责抵消平移/旋转惯性耦合。力或力矩超限时必须
        # 对整个 wrench 等比例缩放；分块裁剪会破坏该比例并重新引入轴间串扰。
        F_fb = Lambda @ np.concatenate([ddx_pos, ddx_ori])
        f3 = F_fb[:3]
        t3 = F_fb[3:]
        nf = np.sqrt(f3 @ f3)
        nt = np.sqrt(t3 @ t3)
        wrench_scale = 1.0
        if nf > self.osc_max_force_fb:
            wrench_scale = min(wrench_scale, self.osc_max_force_fb / nf)
        if nt > self.osc_max_torque_fb:
            wrench_scale = min(wrench_scale, self.osc_max_torque_fb / nt)
        F_fb *= wrench_scale
        tau_task = J.T @ F_fb
        # 零空间姿态稳定（拉向 home，阻尼关节速度）
        tau0 = self.osc_null_kp * (self.osc_q_home - q) - self.osc_null_kd * qd
        N = self._eye_arm - Jbar @ J
        # MuJoCo 把关节被动阻尼放在 qfrc_passive（-d·qd），不包含在 qfrc_bias 中。
        # 补偿它，避免大阻尼模型的旋转通过雅可比耦合为末端位置漂移。
        tau = tau_task + N.T @ tau0 + d.qfrc_bias[:n] + m.dof_damping[:n] * qd

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

    def update(self, dst_pos, dst_quat, admittance_ratio=10, admittance_sub_steps=10):
        """OSC 移动到目标位姿：笛卡尔斜坡 + 导纳偏移 + 每 1ms 力矩控制"""
        dt = self.model.opt.timestep * admittance_sub_steps

        start_pos = self.data.site_xpos[self.eef_site_id].copy()
        start_quat = rotmat_to_quat(self.data.site_xmat[self.eef_site_id].reshape(3, 3))
        T_total = admittance_ratio * dt
        vel_ff = np.zeros(6)
        vel_ff[:3] = (dst_pos - start_pos) / T_total               # 斜坡线速度前馈
        R_start = _quat_to_rotmat_fast(start_quat)
        R_dst = _quat_to_rotmat_fast(dst_quat)
        vel_ff[3:] = (self.osc_ori_ff_scale
                      * _rotmat_to_rotvec_fast(R_dst @ R_start.T) / T_total)

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
