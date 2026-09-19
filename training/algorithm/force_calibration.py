import numpy as np
from algorithm.filter import LowPassFilter

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
        """更新校准力和力矩的估计值
        inertial_ft: 工具惯性力/力矩（传感器系，关于传感器原点），
        由调用方用 mj_objectAcceleration + Newton-Euler 算好传入；
        补偿后运动中无接触时读数应≈0（标准处理链第 3 层：惯性补偿）。"""
        tool_gs = R_ws.T @ self.tool_gw
        f_in = np.zeros(3) if inertial_ft is None else inertial_ft[:3]
        t_in = np.zeros(3) if inertial_ft is None else inertial_ft[3:]

        f_s = -f_ext[:3] - tool_gs + f_in
        f_t = R_ts @ f_s
        
        t_s = -f_ext[3:] - _cross(P_stm, tool_gs) + t_in
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

        # 滤除高频噪声
        self.force = self.force_filter.filter(force, dt=dt)
        self.torque = self.torque_filter.filter(torque, dt=dt)

    def reset(self):
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.force_filter.reset()
        self.torque_filter.reset()
