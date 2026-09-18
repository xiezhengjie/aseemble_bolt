import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from utils.math_utils import *

try:
    from trac_ik import TracIK
    TRAC_IK_AVAILABLE = True
except ImportError:
    TRAC_IK_AVAILABLE = False

class UR5eIK:
    def __init__(self, urdf_filename="/home/xzj/workdir/ROS/ros/ur5e/ur5e_assemble_ws/src/training/urdf/ur5e_assemble.urdf",
                  eef_farme_name="tcp_link", 
                  arm_dof=6,
                  max_iterations = 100, # 最大迭代次数，防止算法陷入无限循环
                  eps = 1e-4,           # 收敛阈值，当误差小于该值时认为算法收敛
                  dt = 2e-2,            # 积分步长，用于更新关节角度
                  damp = 1e-12,         # 阻尼因子，用于避免矩阵奇异
                  verbose=True):
        # pinochio模型加载和初始化
        self.model = pin.buildModelFromUrdf(urdf_filename)
        self.data = self.model.createData()
        # IK 后端选择：默认 pinocchio（确定性局部 IK，保证实验可复现）；
        # 设 UR5E_IK_BACKEND=trac 可恢复旧行为（TracIK Speed 模式 + wall-clock 超时，非确定）。
        self.ik_backend = os.environ.get("UR5E_IK_BACKEND", "pin").lower()
        # trac_ik模型加载和初始化（Windows 等平台可能缺失该包）
        if self.ik_backend == "trac" and TRAC_IK_AVAILABLE:
            self.slover = TracIK(
                base_link_name="base_link",
                tip_link_name=eef_farme_name,
                urdf_path=urdf_filename
            )
        else:
            self.slover = None
            if verbose and self.ik_backend == "trac" and not TRAC_IK_AVAILABLE:
                print("[UR5eIK] trac_ik 不可用，回退 pinocchio")
                self.ik_backend = "pin"
        self.eef_farme_name = eef_farme_name
        self.arm_dof = arm_dof
        self.verbose = verbose

        self.max_iterations = max_iterations 
        self.eps = eps 
        self.dt = dt 
        self.damp = damp 

        self.eef_farme_id = self.model.getFrameId(eef_farme_name)
        if self.eef_farme_id == self.model.nframes:
            raise ValueError(f"Frame tcp_link 不存在，请检查URDF")
    
    def ik(self, tgt_pos: np.ndarray, tgt_quat: np.ndarray, seed_jnt_values: np.ndarray):
        if self.ik_backend == "trac" and self.slover is not None:
            return self.ik_trac(tgt_pos, tgt_quat, seed_jnt_values)
        else:
            return self.ik_pinocchio(tgt_pos, tgt_quat, seed_jnt_values)
    
    def ik_pinocchio(self, tgt_pos: np.ndarray, tgt_quat: np.ndarray, seed_jnt_values: np.ndarray):
        # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
        target_rot = quat_to_rotmat(tgt_quat)
        oMdes = pin.SE3(target_rot, np.array(tgt_pos))
        # 将当前关节角度赋值给变量 q，作为迭代的初始值
        q = seed_jnt_values.copy()

        # 初始化迭代次数为 0
        i = 0
        while True:
            # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            # 计算目标位姿到当前位姿之间的变换
            iMd = self.data.oMf[self.eef_farme_id].actInv(oMdes)
            # 通过李群对数映射将变换矩阵转换为 6 维误差向量（包含位置误差和方向误差），用于量化当前位姿与目标位姿的差异
            err = pin.log(iMd).vector

            # 判断误差是否小于收敛阈值，如果是则认为算法收敛
            if np.linalg.norm(err) < self.eps:
                success = True
                break
            # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
            if i >= self.max_iterations:
                success = False
                break

            # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系
            J = pin.computeFrameJacobian(self.model, self.data, q, self.eef_farme_id, pin.ReferenceFrame.LOCAL)
            # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            # 使用阻尼最小二乘法求解关节速度
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.damp * np.eye(6), err))
            # 根据关节速度更新关节角度
            q = pin.integrate(self.model, q, v * self.dt)
            # 关节限位
            q = np.clip(q, 
                    self.model.lowerPositionLimit[:self.arm_dof],
                    self.model.upperPositionLimit[:self.arm_dof])
            # 每迭代 10 次打印一次当前的误差信息
            if (not i % 10) and self.verbose:
                print(f"{i}: error = {err.T}")
            # 迭代次数加 1
            i += 1

        # 根据算法是否收敛输出相应的信息
        if success:
            if self.verbose: print("Convergence achieved!")
        elif self.verbose:
            print(
                "\n"
                "Warning: the iterative algorithm has not reached convergence "
                "to the desired precision"
            )
        if self.verbose:
            # 打印最终的关节角度和误差向量
            print(f"\nresult: {q.flatten().tolist()}")
            print(f"\nfinal error: {err.T}")
        # 返回最终的关节角度向量
        return q.flatten()
    
    def ik_trac(self, tgt_pos: np.ndarray, tgt_quat: np.ndarray, seed_jnt_values: np.ndarray):
        """使用TracIK求解"""
        return self.slover.ik(tgt_pos, quat_to_rotmat(tgt_quat), seed_jnt_values=seed_jnt_values)

    def ik_mujoco(self,  model, data, tgt_pos: np.ndarray, tgt_quat: np.ndarray, seed_jnt_values: np.ndarray, max_iter:int=100):
        """使用MuJoCo原生IK求解"""
        q = seed_jnt_values.copy()
        target_rot = quat_to_rotmat(tgt_quat)
        eef_site_id = data.site("tcp_site").id
        for _ in range(max_iter):
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            
            # 获取当前末端位姿
            current_pos = data.site_xpos[eef_site_id].copy()
            current_rot = data.site_xmat[eef_site_id].reshape(3, 3)
            
            # 计算位置误差
            pos_err = tgt_pos - current_pos
            
            # 计算旋转误差（使用李代数）
            rot_err = (target_rot @ current_rot.T - np.eye(3))
            rot_err_vec = np.array([rot_err[2,1], rot_err[0,2], rot_err[1,0]])
            
            # 组合误差
            err = np.concatenate([pos_err, rot_err_vec])
            
            if np.linalg.norm(err) < self.eps:
                break
            
            # 获取雅可比矩阵
            J_pos = np.zeros((3, model.nv))
            J_rot = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, J_pos, J_rot, self.eef_site_id)
            J = np.vstack([J_pos[:, :6], J_rot[:, :6]])
            
            # 阻尼最小二乘
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
            q = np.clip(q, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
        
        return q
    

    def fk(self, jnt_values: np.ndarray):
        if self.ik_backend == "trac" and self.slover is not None:
            return self.fk_trac(jnt_values)
        else:
            return self.fk_pinocchio(jnt_values)

    def fk_pinocchio(self, jnt_values: np.ndarray):
        """使用Pinocchio求解正运动学，输入关节角度，输出末端位姿"""
        pin.forwardKinematics(self.model, self.data, jnt_values)
        pin.updateFramePlacements(self.model, self.data)
        frame_placement = self.data.oMf[self.eef_farme_id]
        pos = frame_placement.translation.copy()
        rot = frame_placement.rotation.copy()
        quat = rotmat_to_quat(rot)
        return pos, quat

    def fk_trac(self, jnt_values: np.ndarray):
        """使用TracIK求解正运动学，输入关节角度，输出末端位姿"""
        pos, rotation_matrix = self.slover.fk(jnt_values)
        quat = rotmat_to_quat(rotation_matrix)
        return pos, quat
    
    def fk_mujoco(self, model, data, jnt_values: np.ndarray):
        """正运动学，输入关节角度，输出末端位姿"""
        q_backup = data.qpos[:6].copy()
        data.qpos[:6] = jnt_values.copy()
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[self.eef_site_id].copy()
        rot = data.site_xmat[self.eef_site_id].reshape(3, 3)
        quat = rotmat_to_quat(rot)
        data.qpos[:6] = q_backup.copy()
        mujoco.mj_forward(model, data)
        return pos, quat
    
    def get_eef_pos(self, data):
        eef_site_id = data.site("tcp_site").id
        pos = data.site_xpos[eef_site_id].copy()
        rot = data.site_xmat[eef_site_id].reshape(3, 3)
        quat = rotmat_to_quat(rot)
        return pos, quat

if __name__ == '__main__':
    xml_path="/home/xzj/workdir/ROS/ros/ur5e/ur5e_assemble_ws/src/training/mjcf/ur5e_assemble_sence.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)    
    data = mujoco.MjData(model)    

#    # 打印所有 body 的 pos 和 xpos
#     print("=== Body positions ===")
#     for i in range(model.nbody):
#         name = model.body(i).name
#         pos = model.body_pos[i]
#         xpos = data.xpos[i]
#         print(f"body {i:2d} '{name:20s}': pos={pos}, xpos={xpos}")

#     print(" === Joint positions ===")
#     for i in range(model.njnt):
#         name = model.joint(i).name
#         jnt_pos = model.jnt_pos[i]
#         print(f"jnt {i:2d} '{name:25s}': pos={jnt_pos}")

#     print("=== Sites ===")
#     for i in range(model.nsite):
#         name = model.site(i).name
#         site_pos = model.site_pos[i]
#         site_xpos = data.site_xpos[i]
#         print(f"site {i:2d} '{name:20s}': pos={site_pos}, xpos={site_xpos}")

    eef_site_id = model.site('tcp_site').id
    ur5e_ik = UR5eIK()
    q0 = np.radians([193.37,-106.87,-103.69,-59.44,90.0,13.37])
    data.qpos[:6] = q0
    data.ctrl[:6] = q0
    mujoco.mj_forward(model, data)
    eef_euler = rotmat_to_euler(data.site_xmat[eef_site_id].reshape(3, 3))
    print(f"末端姿态:{np.degrees(eef_euler)}, 末端位置:{data.site_xpos[model.site('tcp_site').id]}") 
    # dest_pos =  np.array([0.5764,0,0.16818]) 
    # dest_euler = normalize_euler(np.radians([180,0,-90]))
    # dest_quat = euler_to_quat(dest_euler)
    # qpos = ur5e_ik.solve_analytical(dest_pos=dest_pos, dest_quat=dest_quat, q_seed=q0, check=True)
    # print("求解结果:", np.degrees(qpos) if qpos is not None else None)

    dest_pos = data.site_xpos[eef_site_id].copy()
    dest_rot = data.site_xmat[eef_site_id].reshape(3, 3)
    dest_quat = rotmat_to_quat(dest_rot)

    print(f"目标位置: {dest_pos}")

    # qpos = ur5e_ik.solve_analytical(dest_pos=dest_pos, dest_quat=dest_quat, q_seed=q0, check=True)
    qpos = ur5e_ik.inverse_kinematics(current_q=q0, target_pos=dest_pos, target_quat=dest_quat)
    print("求解结果:", np.degrees(qpos) if qpos is not None else None)
    print("原始关节:", np.degrees(q0))
        