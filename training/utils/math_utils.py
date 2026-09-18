import numpy as np
from scipy.spatial.transform import Rotation as R

class MathConst:
    @classmethod
    @property
    def EPS(cls) -> float:
        return 1.0e-6

    @classmethod
    @property
    def ERROR(cls) -> float:
        return 1.0e-3

def normalize_euler(euler_rad):
    """ 将欧拉角归一化到 [-π, π) 范围 """
    rot = R.from_euler('xyz', euler_rad)
    normalized = rot.as_euler('xyz')
    return normalized
    
def rotmat_to_euler(rotmat):
    """
    旋转矩阵转欧拉角
    
    参数：
        rotmat: 3x3旋转矩阵
    
    返回：
        euler: [roll, pitch, yaw] 欧拉角（弧度）
    """
    rot = R.from_matrix(rotmat)
    return rot.as_euler('xyz')

def rotmat_to_quat(rotmat):
    """
    旋转矩阵转四元数
    
    参数：
        rotmat: 3x3旋转矩阵

    返回：
        quat: [w, x, y, z] 四元数   
    """
    rot = R.from_matrix(rotmat)
    quat_xyzw = rot.as_quat()  # scipy返回[x,y,z,w]格式
    return np.roll(quat_xyzw, 1)  # 转换为[w,x,y,z]
    
def euler_to_quat(euler):
    """
    欧拉角转四元数
    
    参数：
        euler: [roll, pitch, yaw] 欧拉角（弧度）
    
    返回：
        quat: [w, x, y, z] 四元数
    """
    rot = R.from_euler('xyz', euler)
    quat_xyzw = rot.as_quat()  # scipy返回[x,y,z,w]格式
    return np.roll(quat_xyzw, 1)  # 转换为[w,x,y,z]

def quat_to_euler(quat):
    """
    四元数转欧拉角
    
    参数：
        quat: [w, x, y, z] 四元数
    返回：
        euler: [roll, pitch, yaw] 欧拉角（弧度）
    """
    quat_xyzw = np.roll(quat, -1)  # 转换为[x,y,z,w]格式
    rot = R.from_quat(quat_xyzw)
    return rot.as_euler('xyz')

def quat_to_rotmat(quat):
    """
    四元数转旋转矩阵
    
    参数：
        quat: [w, x, y, z] 四元数
    
    返回：
        rotmat: 3x3旋转矩阵
    """
    quat_xyzw = np.roll(quat, -1)  # 转换为[x,y,z,w]格式
    rot = R.from_quat(quat_xyzw)
    return rot.as_matrix()

def quat_error_angle(q1, q2):
    """
    计算两个四元数之间的旋转角误差（弧度）
    q1, q2: [w, x, y, z] 或 [x, y, z, w]
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0.0, 1.0)
    return 2 * np.arccos(dot)

def quat_multiply(q1, q2):
    """
    四元数乘法 q1 * q2
    
    参数：
        q1, q2: 四元数 [w, x, y, z]
    
    返回：
        result: 四元数乘积 [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.result_type(q1, q2))


class MathUtils:
    @staticmethod
    def near_zero(value: float) -> bool:
        return np.abs(value) < MathConst.EPS

    @staticmethod
    def _as_vec(x, dim):
        a = np.atleast_1d(np.asarray(x, np.float64))
        if a.size == 1:
            return np.full(dim, a.item())
        if a.size != dim:
            raise ValueError(f"增益长度应为 {dim}，得到 {a.size}")
        return a