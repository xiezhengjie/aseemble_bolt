import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
import logging
import numpy as np
import time
from utils import rl_utils
from algorithm.filter import LowPassFilter
from envs.assemble_mujoco_env import AssembleMuJoCoEnv

class DataRecorder:
    COOLDOWN_SEC = 0.5
    DEADZONE = 0.1
    TOLERANCE = 5e-4
    MAX_FINAL_YAW_DEG = 10.0  # 完成时 |yaw| 超过该角度的 episode 丢弃
    def __init__(self, 
                 xml_path, 
                 urdf_path, save_data_path):
        self.xml_path = xml_path
        self.urdf_path = urdf_path
        self.save_data_path = save_data_path
        self.buttonCooldown = 0.0
        self.is_recording = False
        self.record_buffer = [[[],[],[]]]
        self.action_zero = np.zeros(6)
        self.move_mode = 0 # 0-快速,缩放1 1-中速,缩放2 2-慢速,缩放3
        self.ctrl_mode = 0 # 0-手柄完全控制 1-自由控制
        # self.random_delta = np.array([
        #             np.random.uniform(-0.001, 0.001),   # X: ±3mm
        #             np.random.uniform(-0.001, 0.001),   # Y: ±3mm
        #             np.random.uniform(-0.003, 0.003),   # Z: ±3mm 
        #             np.random.uniform(-5, 5),           # yaw: ±5°
        #             np.random.uniform(-0.003, 0.003),        # X: ±0mm
        #             np.random.uniform(-0.003, 0.003),        # Y: ±0mm
        #             np.random.uniform(-0.003, 0.003),        # Z: ±0mm 
        #             np.random.uniform(-15, 15),         # Roll: ±15°
        #             np.random.uniform(-15, 15),         # Pitch: ±15°
        #             np.random.uniform(-30, 30)             # Yaw: ±0°
        #         ])
        self.random_delta = np.array([
                        np.random.uniform(-0.001, 0.001),   # X: ±3mm
                        np.random.uniform(-0.001, 0.001),   # Y: ±3mm
                        np.random.uniform(-0.003, 0.003),   # Z: ±3mm 
                        np.random.uniform(-5, 5),           # yaw: ±5°
                        np.random.uniform(-0.002, 0.002),        # X: ±0mm
                        np.random.uniform(-0.002, 0.002),        # Y: ±0mm
                        np.random.uniform(-0.002, 0.002),        # Z: ±0mm 
                        np.random.uniform(-10, 10),         # Roll: ±10°
                        np.random.uniform(-10, 10),         # Pitch: ±10°
                        np.random.uniform(-15, 15)             # Yaw: ±15°
                    ])
        self.filter = LowPassFilter(cutoff_freq=40, dt=0.1)

        
        # 创建AssembleMuJoCoEnv环境（录制始终存单帧，训练时 FrameStack 再叠）
        self.env = AssembleMuJoCoEnv(xml_path=xml_path,
                                     urdf_path=urdf_path,
                                     render_mode="human",
                                     is_use_force_control=True,
                                     admittance_m=6.0,
                                     admittance_j=0.6,
                                     admittance_k_t=np.array([1800, 1800, 3000]),
                                     admittance_k_r=np.array([12, 6, 20]),
                                     admittance_zeta_t=2.2,
                                     admittance_zeta_r=1.2,
                                    #  admittance_b_r=400.0,
                                    #  admittance_b_t=400.0,
                                     admittance_force_deadzone=0.2,
                                     admittance_torque_deadzone=0.01,
                                     )
        # self.env.ur5e_controller.admittance_controller.set_axis_mask([0, 0, 0, 1, 0, 0])

        # 环境复位
        self.env.reset(options={'random_delta': self.random_delta})

        # 配置 logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)

        # 初始化硬件输入
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        if self.joystick: self.joystick.init()
        else:
            self.logger.error("未检测到手柄")
            exit()

    def record_toggle(self):
        if self.is_recording:
            self.is_recording = False
            self.logger.info("录制停止。")
        else:
            self.logger.info("录制开始 (Recording ON)")
            self.record_buffer[-1] = [[],[],[]]
            self.is_recording = True

    def reset_toggle(self):
        self.random_delta = np.array([
            np.random.uniform(-0.001, 0.001),   # X: ±3mm
            np.random.uniform(-0.001, 0.001),   # Y: ±3mm
            np.random.uniform(-0.003, 0.003),   # Z: ±3mm 
            np.random.uniform(-5, 5),           # yaw: ±5°
            np.random.uniform(-0.00, 0.00),        # X: ±0mm
            np.random.uniform(-0.00, 0.00),        # Y: ±0mm
            np.random.uniform(-0.00, 0.00),        # Z: ±0mm 
            np.random.uniform(-10, 10),         # Roll: ±10°
            np.random.uniform(-10, 10),         # Pitch: ±10°
            np.random.uniform(0, 0)             # Yaw: ±0°
        ])
        # self.random_delta = np.array([
        #         np.random.uniform(-0.001, 0.001),   # X: ±3mm
        #         np.random.uniform(-0.001, 0.001),   # Y: ±3mm
        #         np.random.uniform(-0.003, 0.003),   # Z: ±3mm 
        #         np.random.uniform(-5, 5),           # yaw: ±5°
        #         np.random.uniform(-0.003, 0.003),        # X: ±0mm
        #         np.random.uniform(-0.003, 0.003),        # Y: ±0mm
        #         np.random.uniform(-0.003, 0.003),        # Z: ±0mm 
        #         np.random.uniform(-15, 15),         # Roll: ±15°
        #         np.random.uniform(-15, 15),         # Pitch: ±15°
        #         np.random.uniform(-30, 30)             # Yaw: ±0°
        #     ])
        self.logger.info(("="*7)+"随机重置已更新"+("="*7))
        
    def joystick_control(self):
        if not self.joystick: return False
        pygame.event.pump()

        # 录环境归一化后的 10 维单帧（与 reset/step 观测一致）；stack 在 load_data 时转换
        observation = self.env._get_observation()
        now = time.time()
        # # A 键 (0): 当前条录制完成
        # if self.joystick.get_button(0) and (now - self.buttonCooldown > self.COOLDOWN_SEC):
        #     self.buttonCooldown = now
        #     self.env.reset(options={'random_delta': self.random_delta})
        #     self.move_mode = 0
        #     self.filter.reset()
        #     if self.is_recording:
        #         self.record_buffer.append([[],[],[]])
        #         self.logger.info(f"当前条录制完成。开始录制第{len(self.record_buffer)}条录制")
        # B 键 (1)：切换控制模型
        if  self.joystick.get_button(1) and (now - self.buttonCooldown > self.COOLDOWN_SEC):
            self.buttonCooldown = now
            self.ctrl_mode = (self.ctrl_mode + 1) % 2
            self.env.reset(options={'random_delta': self.random_delta})
            self.move_mode = 0
            self.filter.reset()
            self.logger.info(f"切换控制模型为 {self.ctrl_mode} (0-手柄完全控制, 1-自由控制)")
        # X 键 (2): 切换移动模式
        if self.joystick.get_button(2) and (now - self.buttonCooldown > self.COOLDOWN_SEC):
            self.buttonCooldown = now
            self.move_mode = (self.move_mode + 1) % 3
            self.logger.info(f"切换移动模式为 {self.move_mode} (0-快速 1.5mm/step, 1-中速 0.75mm/step, 2-慢速 0.5mm/step)")
        # Y 键 (3): 随机重置并更新
        if self.joystick.get_button(3) and (now - self.buttonCooldown > self.COOLDOWN_SEC):
            self.buttonCooldown = now
            self.filter.reset()
            self.reset_toggle()
            self.env.reset(options={'random_delta': self.random_delta})
            self.move_mode = 0
            if self.is_recording:
                self.record_buffer[-1] = [[],[],[]]
        # back 键 (6): 退出当前条录制
        if self.is_recording and self.joystick.get_button(6) and (now - self.buttonCooldown > self.COOLDOWN_SEC):
            self.buttonCooldown = now
            self.filter.reset()
            self.env.reset(options={'random_delta': self.random_delta})
            self.move_mode = 0
            if self.is_recording:
                self.record_buffer[-1] = [[],[],[]]
            self.logger.info(f"当前条录制退出。{len(self.record_buffer)-1}条录制完成")
        # start 键 (7): 录制开关
        if self.joystick.get_button(7) and (now - self.buttonCooldown > self.COOLDOWN_SEC):
            self.buttonCooldown = now
            self.env.reset(options={'random_delta': self.random_delta})
            self.move_mode = 0
            self.filter.reset()
            self.record_toggle()

        # 位置控制 (左摇杆)
        ax0, ax1, ax2 = self.joystick.get_axis(0), self.joystick.get_axis(1), self.joystick.get_axis(2)
        ax2 = (1 + ax2) / 2 # 将轴范围从 [-1, 1] 映射到 [0, 1]  
        dx = -(abs(ax1) > self.DEADZONE) * ax1 
        dy = -(abs(ax0) > self.DEADZONE) * ax0 
        dz = (abs(ax2) > self.DEADZONE) * (2* self.joystick.get_button(4) - 1) * ax2   # LB 控制 Z 轴上下 
        delta_pos = np.array([dx, dy, dz])

        # 旋转控制 (右摇杆)
        ax3, ax4, ax5 = self.joystick.get_axis(3), self.joystick.get_axis(4), self.joystick.get_axis(5)
        ax5 = (1 + ax5) / 2  
        dr_x = (abs(ax3) > self.DEADZONE) * ax3 
        dr_y = -(abs(ax4) > self.DEADZONE) * ax4 
        dr_z = (abs(ax5) > self.DEADZONE) * (1 - 2* self.joystick.get_button(5)) * ax5 # RB 控制 Z 轴旋转方向
        delta_euler = np.array([dr_x, dr_y, dr_z])

        action = np.concatenate([delta_pos / (self.move_mode + 1), delta_euler / (self.move_mode + 1)])
        
        if self.ctrl_mode == 0:
           if not np.allclose(self.action_zero, action, rtol=0, atol=self.TOLERANCE): # 当有动作变化时才运动
                # 应用低通滤波器, 以减少动作的抖动
                action = self.filter.filter(action)
                if not np.allclose(self.action_zero, action, rtol=0, atol=self.TOLERANCE): # 当有动作变化时才运动
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    if self.is_recording:
                        self.logger.info(f"记录动作: {action}, 当前步数: {self.env.current_step}, 深度: {info['depth']}, 状态: {info['state']}, \
                                         力: {info['force']}, 力矩: {info['torque']}, 位置误差xy: {info['position_error_xy']}, 偏角: {info['angle_z']}, yaw误差: {info['yaw_error']}")
                        self.record_buffer[-1][0].append(observation)
                        self.record_buffer[-1][1].append(action)
                        self.record_buffer[-1][2].append(terminated or truncated)

                        done = info['success']
                        if terminated and not done:
                            self.env.reset(options={'random_delta': self.random_delta})
                            self.filter.reset()
                            self.move_mode = 0
                            self.record_buffer[-1] = [[],[],[]]
                            self.logger.info(f"当前装配任务失败，录制退出，{len(self.record_buffer)-1}条录制完成，输出信息为：{info}")
                        elif truncated:
                            self.env.reset(options={'random_delta': self.random_delta})
                            self.filter.reset()
                            self.move_mode = 0
                            self.record_buffer[-1] = [[],[],[]]
                            self.logger.info(f"当前装配任务被截断，录制退出，{len(self.record_buffer)-1}条录制完成，输出信息为：{info}")
                        elif done:
                            final_yaw = self.record_buffer[-1][0][-1][3]
                            final_yaw_deg = np.degrees(final_yaw)
                            self.env.reset(options={'random_delta': self.random_delta})
                            self.filter.reset()
                            self.move_mode = 0
                            if abs(final_yaw_deg) > self.MAX_FINAL_YAW_DEG:
                                self.record_buffer[-1] = [[],[],[]]
                                self.logger.warning(f"最终yaw={final_yaw_deg:.1f}°超过阈值{self.MAX_FINAL_YAW_DEG:.0f}°，丢弃该条录制，{len(self.record_buffer)-1}条录制完成")
                            else:
                                self.record_buffer.append([[],[],[]])
                                self.logger.info(f"当前装配任务完成，完成录制第{len(self.record_buffer)-1}条录制，最终yaw={final_yaw_deg:.1f}°，输出信息为：{info}")
        elif self.ctrl_mode == 1: 
            obs, reward, terminated, truncated, info = self.env.step(action)
        
        return self.joystick.get_button(8) # Start 键退出


    def recode_run(self):
        try:
            while True:
                if self.joystick_control(): 
                    self.logger.info("录制中断。") 
                    self.record_buffer[-1] = [[],[],[]] # 录制中断时，丢弃当前条数据
                    break
        except KeyboardInterrupt:
            if self.is_recording:
                self.logger.info("录制中断。") 
                self.record_buffer[-1] = [[],[],[]] # 录制中断时，丢弃当前条数据
        finally:
            # 条件判断
            if self.is_recording and len(self.record_buffer[0][0]) > 0:
                max_yaw_rad = np.radians(self.MAX_FINAL_YAW_DEG)
                record_len = 0
                # 本次新数据
                new_states, new_actions, new_dones= [], [], []   
                for record in self.record_buffer:
                    if len(record[0]) > 0:
                        if record[2][-1] == True:
                            final_yaw = record[0][-1][3]
                            if abs(final_yaw) > max_yaw_rad:
                                self.logger.warning(f"保存时过滤：最终yaw={np.degrees(final_yaw):.1f}°超过阈值{self.MAX_FINAL_YAW_DEG:.0f}°，丢弃该条")
                                continue
                            record_len += 1
                            new_states.append(np.array(record[0], dtype=np.float32))
                            new_actions.append(np.array(record[1], dtype=np.float32))
                            new_dones.append(np.array(record[2], dtype=np.float32))

                # 追加已有数据（如果存在），同样按最终 yaw 过滤
                if os.path.exists(self.save_data_path):
                    old = np.load(self.save_data_path, allow_pickle=True)
                    old_s, old_a, old_d = old['states'], old['actions'], old['dones']
                    ep_ends = np.where(old_d > 0.5)[0]
                    if len(ep_ends) > 0:
                        ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
                        kept_old_s, kept_old_a, kept_old_d = [], [], []
                        removed_old = 0
                        for s, e in zip(ep_starts, ep_ends + 1):
                            if abs(old_s[e - 1, 3]) <= max_yaw_rad:
                                kept_old_s.append(old_s[s:e])
                                kept_old_a.append(old_a[s:e])
                                kept_old_d.append(old_d[s:e])
                            else:
                                removed_old += 1
                        if removed_old > 0:
                            self.logger.warning(f"旧数据中过滤掉 {removed_old} 条最终yaw>{self.MAX_FINAL_YAW_DEG:.0f}°的记录")
                        if kept_old_s:
                            new_states.append(np.concatenate(kept_old_s))
                            new_actions.append(np.concatenate(kept_old_a))
                            new_dones.append(np.concatenate(kept_old_d))
                    old.close() 

                # 合并所有数据
                new_states = np.concatenate(new_states)
                new_actions = np.concatenate(new_actions)
                new_dones = np.concatenate(new_dones)

                # 保存
                np.savez(self.save_data_path, states=new_states, actions=new_actions, dones=new_dones)
                self.logger.info(f"录制数据已保存到 {self.save_data_path}，本次共录制 {record_len} 条数据")
            else:
                self.logger.info(f"没有录制到数据，未保存。")
            self.env.close()
            pygame.quit()


if __name__ == "__main__":
    root_dir = rl_utils.find_project_root()
    save_data_dir = root_dir/"datasets"
    xml_path = str(root_dir/"mjcf/ur5e_assemble_sence.xml")
    urdf_path = str(root_dir/"urdf/ur5e_assemble.urdf")
    save_data_path = save_data_dir/"recorded_data.npz"

    recorder = DataRecorder(xml_path, urdf_path, save_data_path)
    recorder.recode_run()