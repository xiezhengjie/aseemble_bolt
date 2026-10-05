import numpy as np
from training.common.filter import LowPassFilter

def _cross(a, b):
    """np.cross 的快速替代（np.cross 的 Python 封装开销 ~80µs，热路径不可接受）"""
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

class ForceCalibrationSim:
    def __init__(self,
                 data,
                 g=np.array([0,0,-9.8]),
                 mass=1,
                 cutoff_freq=100,
                 dt=0.002,
                 force_tr=0, torque_tr=0
                 ) -> None:
        self.data = data
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.tool_gw = g*mass
        self.force_tr = force_tr
        self.torque_tr = torque_tr
        self.force_filter = LowPassFilter(cutoff_freq=cutoff_freq, dt=dt)
        self.torque_filter = LowPassFilter(cutoff_freq=cutoff_freq, dt=dt)

    def update(self, f_ext, R_ws, R_ts, P_stm, P_ts, dt, inertial_ft=None):
        """更新校准力和力矩的估计值。

        滤波顺序固定为：原始传感器力/力矩 -> 一阶低通 -> 重力/惯性/坐标校准
        -> 工具坐标系死区。不能在校准输出上再滤波：校准会改变力矩的数值尺度，
        且旋转、力臂和惯性补偿会把滤波器状态混入物理变换。

        inertial_ft: 工具惯性力/力矩（传感器系，关于传感器原点），
        由调用方用 mj_objectAcceleration + Newton-Euler 算好传入；
        补偿后运动中无接触时读数应≈0（标准处理链第 3 层：惯性补偿）。"""
        f_ext = np.asarray(f_ext, dtype=np.float64).reshape(-1)

        # 传感器原始坐标系下先滤波。力和力矩分别使用独立的一阶滤波器，
        # 避免校准后的小力矩被低通时与重力/力臂补偿耦合。
        force_raw = self.force_filter.filter(f_ext[:3], dt=dt)
        torque_raw = self.torque_filter.filter(f_ext[3:], dt=dt)

        tool_gs = R_ws.T @ self.tool_gw
        f_in = np.zeros(3) if inertial_ft is None else inertial_ft[:3]
        t_in = np.zeros(3) if inertial_ft is None else inertial_ft[3:]

        f_s = -force_raw - tool_gs + f_in
        f_t = R_ts @ f_s
        
        t_s = -torque_raw - _cross(P_stm, tool_gs) + t_in
        t_t = R_ts @ t_s + _cross(P_ts, f_t)

        # 以工具坐标系表示力和力矩
        force = f_t  
        torque = t_t

        # 死区处理
        for i in range(3):
            if abs(force[i]) < self.force_tr:
                force[i] = 0.0
            else:
                force[i] = np.sign(force[i]) * (abs(force[i]) - self.force_tr)
        for i in range(3):
            if abs(torque[i]) < self.torque_tr:
                torque[i] = 0.0
            else:
                torque[i] = np.sign(torque[i]) * (abs(torque[i]) - self.torque_tr)

        # 校准输出不再重复滤波；此处已经完成原始传感器域低通。
        self.force = force.copy()
        self.torque = torque.copy()

    def reset(self):
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.force_filter.reset()
        self.torque_filter.reset()
