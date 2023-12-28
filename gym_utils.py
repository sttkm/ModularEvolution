import os
import gym
import time
import numpy as np
import multiprocessing.pool

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

def make_env(env_id, robot):
    def _init():
        env = gym.make(env_id, body=robot[0], connections=robot[1])
        env.action_space = gym.spaces.Box(low=-1.0, high=1.0,
            shape=env.action_space.shape, dtype=np.float)
        env = Monitor(env, None, allow_early_resets=True)
        return env
    return _init


def make_vec_envs(env_id, robot, num_processes, gamma=None, subproc=True, vecnorm=True):
    envs = [make_env(env_id, robot) for i in range(num_processes)]

    if subproc and num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if vecnorm:
        if gamma is not None:
            envs = VecNormalize(envs, gamma=gamma)
        else:
            envs = VecNormalize(envs, norm_reward=False)
    
    return envs


from stable_baselines3.common.running_mean_std import RunningMeanStd as RunningMeanStd_

class RunningMeanStd(RunningMeanStd_):
    def __init__(self, epsilon: float = 1e-4, shape=()):
        if len(shape)==0:
            self.mean = 0.0
            self.var = 0.0
        else:
            self.mean = np.zeros((1, shape[-1]), np.float64)
            self.var = np.ones((1, shape[-1]), np.float64)
        self.count = epsilon

    def copy(self):
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def update(self, arr: np.ndarray) -> None:
        arr = arr.reshape((-1, arr.shape[-1]))
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = min(new_count, 2**20)


from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class VecNormalize(VecNormalize_):
    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        norm_obs_keys=None,
    ):
        VecNormalize_.__init__(self, venv)

        self.norm_obs = norm_obs
        self.norm_obs_keys = norm_obs_keys
        # Check observation spaces
        if self.norm_obs:
            self._sanity_checks()

            if isinstance(self.observation_space, gym.spaces.Dict):
                self.obs_spaces = self.observation_space.spaces
                self.obs_rms = {key: RunningMeanStd(shape=self.obs_spaces[key].shape) for key in self.norm_obs_keys}
            else:
                self.obs_spaces = None
                self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])
        self.old_reward = np.array([])
