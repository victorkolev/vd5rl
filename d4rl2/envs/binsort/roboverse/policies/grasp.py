import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS


class Grasp:

    def __init__(self, env, pick_height_thresh=-0.23, xyz_action_scale=7.0,
                 pick_point_noise=0.00):
        self.env = env
        self.pick_height_thresh = pick_height_thresh
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point += np.random.normal(scale=self.pick_point_noise, size=(3,))
        self.pick_point[2] = -0.32 + np.random.normal(scale=0.01)

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        # gripper_droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False

        if gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # Hold
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.]

        agent_info = dict(done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info
