import os

import pickle
import numpy as np

from run_ppo import run_ppo


class ppoConfig:
    def __init__(self):
        self.num_processes = 4
        self.steps = 128
        self.num_mini_batch = 4
        self.epochs = 4
        self.learning_rate = 2.5e-4
        self.gamma = 0.99
        self.clip_range = 0.1
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.lr_decay = True
        self.gae_lambda = 0.95
        self.init_log_std = 0.0

class EvogymStructureEvaluator:
    def __init__(self, env_id, save_path, ppo_iters, eval_interval, ppo_config, deterministic=True, resume=False):
        self.env_id = env_id
        self.save_path = save_path
        self.robot_save_path = os.path.join(save_path, 'robot')
        self.controller_save_path = os.path.join(save_path, 'controller')
        self.ppo_iters = ppo_iters
        self.eval_interval = eval_interval
        self.ppo_config = ppo_config
        self.deterministic = deterministic

        if not resume:
            os.makedirs(self.robot_save_path, exist_ok=True)
            os.makedirs(self.controller_save_path, exist_ok=True)

    def evaluate_structure(self, key, robot, generation=None):

        if type(robot) is not tuple:
            if not is_connected(robot):
                return -1
            robot = (robot, get_full_connectivity(robot))

        file_robot = os.path.join(self.robot_save_path, f'{key}')
        file_controller = os.path.join(self.controller_save_path, f'{key}')
        np.savez(file_robot, *robot)

        reward = run_ppo(
            env_id=self.env_id,
            robot=robot,
            train_iters=self.ppo_iters,
            eval_interval=self.eval_interval,
            save_file=file_controller,
            config=self.ppo_config,
            deterministic=self.deterministic
        )

        return reward
    

from evogym import is_connected, has_actuator, hashable, get_full_connectivity

def to_hash(robot):
    hash = ",".join(["".join(map(lambda z: str(int(z)), c)) for c in robot])
    return hash

class EvogymStructureConstraint:
    def __init__(self, decode_function):
        self.decode_function = decode_function
        self.hashes = {}

    def has_actuator(self, body):
        voxel_count = np.sum(body > 0)
        actuator_count = np.sum(body>=3)
        density = actuator_count / voxel_count
        return density > 0.3

    def check_density(self, body):
        voxel_count = np.sum(body>0)
        return voxel_count >= 8

    def eval_constraint(self, genome, config, *args):
        robot = self.decode_function(genome, config)
        body,_ = robot
        validity = is_connected(body) and self.has_actuator(body) and self.check_density(body)
        if validity:
            robot_hash = to_hash(body)
            if robot_hash in self.hashes:
                validity = False
            else:
                self.hashes[robot_hash] = True

        return validity
    
    def save_hashes(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.hashes, f)

    def load_hashes(self, file):
        with open(file, "rb") as f:
            self.hashes = pickle.load(f)