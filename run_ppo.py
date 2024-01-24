import os
import csv
import time
import numpy as np
import torch

from ppo import Policy, PPO

from gym_utils import make_vec_envs

def evaluate(policy, envs, num_eval=1, deterministic=False):

    obs = envs.reset()
    episode_rewards = []
    while len(episode_rewards) < num_eval:
        with torch.no_grad():
            action = policy.predict(obs, deterministic=deterministic)
        obs, _, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info:
                episode_rewards.append(info['episode']['r'])
    return np.mean(episode_rewards)


def run_ppo(env_id, robot, train_iters, eval_interval, save_file, config=None, deterministic=False):

    train_envs = make_vec_envs(env_id, robot, config.num_processes, gamma=config.gamma, subproc=False)

    eval_envs = make_vec_envs(env_id, robot, 1, gamma=None, subproc=False)
    eval_envs.training = False

    policy = Policy(
        train_envs.observation_space,
        train_envs.action_space,
        init_log_std=config.init_log_std,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)

    algo = PPO(
        train_envs,
        learning_rate=3.5e-4,
        n_steps=config.steps,
        batch_size=config.steps*config.num_processes//config.num_mini_batch,
        n_epochs=config.epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range,
        normalize_advantage=True,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device='cpu',
        lr_decay=config.lr_decay,
        max_iter=train_iters*10)


    max_reward = float('-inf')
    interval = time.time()

    for iter in range(train_iters):
        
        algo.step(policy, optimizer)

        if (iter+1) % eval_interval == 0:
            eval_envs.obs_rms = train_envs.obs_rms.copy()
            reward = evaluate(policy, eval_envs, num_eval=1, deterministic=deterministic)
            if reward > max_reward:
                max_reward = reward
                torch.save([policy.state_dict(), train_envs.obs_rms], save_file + '.pt')

            now = time.time()
            log_std = policy.log_std.mean()
            # print(f'iteration: {iter+1:=5}  elapsed times: {now-interval:.3f}  reward: {reward:6.3f}  log_std: {log_std:.5f}')

    del train_envs
    del eval_envs

    return max_reward
