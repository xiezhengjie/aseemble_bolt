"""OSC 位姿跟踪回归：旋转速度、平移串扰和接触旋转。

用法: /usr/bin/python3.10 training/tests/osc_pose_tracking_test.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.assemble_mujoco_env import AssembleMuJoCoEnv
from utils.math_utils import (
    quat_conj,
    quat_multiply,
    quat_to_rotvec,
    rotmat_to_quat,
)


ROOT = Path(__file__).resolve().parents[2]
XML = str(ROOT / "mjcf" / "ur5e_assemble_sence.xml")
URDF = str(ROOT / "urdf" / "ur5e_assemble.urdf")


def make_env(force_control=False):
    env = AssembleMuJoCoEnv(
        xml_path=XML,
        urdf_path=URDF,
        render_mode=None,
        is_use_force_control=force_control,
        max_episodic_steps=500,
    )
    env.reset(seed=1234)
    return env


def current_pose(env):
    site = env.eef_site_id
    pos = env.data.site_xpos[site].copy()
    quat = rotmat_to_quat(env.data.site_xmat[site].reshape(3, 3))
    return pos, quat


def relative_rotvec(q1, q0):
    dq = quat_multiply(q1, quat_conj(q0))
    if dq[0] < 0.0:
        dq = -dq
    return quat_to_rotvec(dq)


def check_action_uses_current_pose():
    env = make_env()
    try:
        pos, quat = current_pose(env)
        env.last_pos = pos + np.array([0.1, -0.1, 0.1])
        env.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
        action = np.array([0.5, -0.25, 0.0, 0.0, 0.0, 0.2])
        target_pos, _ = env._action_to_pose(action)
        np.testing.assert_allclose(
            target_pos, pos + action[:3] * env.pos_action_scale, atol=1e-12
        )

        env.last_pos = pos + np.array([-0.1, 0.1, -0.1])
        zero_pos, zero_quat = env._action_to_pose(np.zeros(6))
        np.testing.assert_allclose(zero_pos, pos, atol=1e-12)
        assert np.abs(np.dot(zero_quat, quat)) > 1.0 - 1e-12
    finally:
        env.close()


def check_free_space_rotation():
    expected = 40 * 0.01
    results = []
    for axis in range(3):
        env = make_env()
        try:
            pos0, quat0 = current_pose(env)
            action = np.zeros(6)
            action[3 + axis] = 1.0
            for _ in range(40):
                env.step(action)
            pos1, quat1 = current_pose(env)
            rotvec = relative_rotvec(quat1, quat0)
            drift = pos1 - pos0

            assert abs(rotvec[axis] - expected) < 0.01
            assert np.linalg.norm(np.delete(rotvec, axis)) < 0.005
            assert np.max(np.abs(drift)) < 0.00015
            results.append((axis, rotvec.copy(), drift.copy()))
        finally:
            env.close()
    return results


def check_free_space_translation():
    expected = 15 * 0.002
    results = []
    for axis in range(3):
        env = make_env()
        try:
            pos0, quat0 = current_pose(env)
            action = np.zeros(6)
            action[axis] = 1.0
            for _ in range(15):
                env.step(action)
            pos1, quat1 = current_pose(env)
            delta = pos1 - pos0
            rotvec = relative_rotvec(quat1, quat0)

            assert abs(delta[axis] - expected) < 0.0005
            assert np.linalg.norm(np.delete(delta, axis)) < 0.0001
            assert np.linalg.norm(rotvec) < 0.005
            results.append((axis, delta.copy(), rotvec.copy()))
        finally:
            env.close()
    return results


def check_contact_rotation_feedforward():
    env = make_env(force_control=True)
    try:
        max_force = 0.0
        for _ in range(55):
            _, _, _, _, info = env.step(np.array([0, 0, -1.0, 0, 0, 0]))
            max_force = max(max_force, float(np.linalg.norm(info["force"])))

        accumulated_yaw = 0.0
        action = np.array([0, 0, -0.15, 0, 0, 1.0])
        for _ in range(40):
            _, quat0 = current_pose(env)
            _, _, _, _, info = env.step(action)
            _, quat1 = current_pose(env)
            accumulated_yaw += relative_rotvec(quat1, quat0)[2]
            max_force = max(max_force, float(np.linalg.norm(info["force"])))

        assert max_force > 2.0
        assert abs(accumulated_yaw - 0.4) < 0.02
        return accumulated_yaw, max_force
    finally:
        env.close()


def check_admittance_contact_moment_direction():
    """偏孔下压时，接触力矩、导纳旋转目标和 TCP 实际转角必须同号。"""
    results = []
    for x_offset, expected_sign in ((0.004, 1.0), (-0.004, -1.0)):
        env = AssembleMuJoCoEnv(
            xml_path=XML,
            urdf_path=URDF,
            render_mode=None,
            is_use_force_control=True,
            max_episodic_steps=500,
        )
        try:
            random_delta = np.zeros(10)
            random_delta[4] = x_offset
            env.reset(seed=0, options={"random_delta": random_delta})

            action = np.array([0.0, 0.0, -0.75, 0.0, 0.0, 0.0])
            contact_quat = None
            contact_steps = 0
            for _ in range(90):
                _, quat_before = current_pose(env)
                _, _, _, _, info = env.step(action)
                if np.linalg.norm(info["force"]) > 2.0:
                    if contact_quat is None:
                        contact_quat = quat_before
                    contact_steps += 1
                    if contact_steps >= 16:
                        break

            assert contact_quat is not None
            _, quat_after = current_pose(env)
            actual_y = relative_rotvec(quat_after, contact_quat)[1]
            dq_y = quat_to_rotvec(env.ur5e_controller.admittance_dq)[1]
            rot = env.data.site_xmat[env.eef_site_id].reshape(3, 3)
            torque_y = (rot @ env.ur5e_controller.calibrated_ft[3:])[1]

            assert expected_sign * torque_y > 0.01
            assert expected_sign * dq_y > 1e-4
            assert expected_sign * actual_y > 1e-4
            results.append((x_offset, torque_y, dq_y, actual_y))
        finally:
            env.close()
    return results


if __name__ == "__main__":
    check_action_uses_current_pose()
    rotation = check_free_space_rotation()
    translation = check_free_space_translation()
    contact_yaw, peak_force = check_contact_rotation_feedforward()
    contact_moment = check_admittance_contact_moment_direction()

    for axis, rotvec, drift in rotation:
        print(
            f"rotation axis={axis}: rotvec={rotvec}, "
            f"position_drift_mm={drift * 1000}"
        )
    for axis, delta, rotvec in translation:
        print(
            f"translation axis={axis}: delta_mm={delta * 1000}, "
            f"rotation_deg={np.degrees(np.linalg.norm(rotvec)):.4f}"
        )
    print(f"contact yaw={contact_yaw:.6f}rad, peak_force={peak_force:.3f}N")
    for x_offset, torque_y, dq_y, actual_y in contact_moment:
        print(
            f"side contact x={x_offset * 1000:+.1f}mm: "
            f"Ty={torque_y:+.5f}Nm, dq_y={np.degrees(dq_y):+.4f}deg, "
            f"actual_y={np.degrees(actual_y):+.4f}deg"
        )
    print("osc pose tracking checks: PASS")
