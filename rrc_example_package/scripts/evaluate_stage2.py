#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys

from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy

import shutil, os
import time
import numpy as np
import torch
from rrc_example_package.her.rl_modules.models import actor

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, clip_obs=200, clip_range=5):
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def main():
    sim=False
    load_ac=True
    num_dice=9
    model_path = '/userhome/acmodel.pt'
    remote_run = not sim
    
    visualization=True
    enable_cameras=True
    scale_dice=1
    step_size=20
    max_steps=int(120000 / step_size)
    distance_threshold=0.02
    include_dice_velocity=False
    include_dice_orient=False
    single_dice_focus=True
    single_goal_focus=True
    choose_strategy=True
    estimate_states=True
    
    if sim:
        goal=None
        # Set up sim env
        env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(
            goal,
            step_size=step_size,
            sim=True,
            remote_run=remote_run,
            #
            visualization=visualization,
            enable_cameras=enable_cameras,
            num_dice=num_dice,
            distance_threshold=distance_threshold,
            max_steps=max_steps,
            include_dice_velocity=include_dice_velocity,
            include_dice_orient=include_dice_orient,
            scale_dice=scale_dice,
            single_dice_focus=single_dice_focus,
            single_goal_focus=single_goal_focus,
            choose_strategy=choose_strategy,
            estimate_states=estimate_states
        )
        print('Sim env created')
    else:
        # the goal is passed as JSON string
        goal_json = sys.argv[1]
        goal = json.loads(goal_json)
        print('goal loaded')
        # Set-up real env
        env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(
            goal,
            step_size=step_size,
            sim=sim,
            visualization=visualization,
            enable_cameras=enable_cameras,
            num_dice=num_dice,
            distance_threshold=distance_threshold,
            max_steps=max_steps,
            include_dice_velocity=include_dice_velocity,
            include_dice_orient=include_dice_orient,
            scale_dice=scale_dice,
            single_dice_focus=single_dice_focus,
            single_goal_focus=single_goal_focus,
            choose_strategy=choose_strategy,
            estimate_states=estimate_states
        )
        print('real env created')
    
    observation = env.reset()
    print('env reset')
    env_params = {'obs': observation['observation'].shape[0],
            'goal': observation['desired_goal'].shape[0],
            'ag': observation['achieved_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    
    if load_ac:
        print('Loading in model from: {}'.format(model_path))
        o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        # create the actor network
        actor_network = actor(env_params)
        actor_network.load_state_dict(model)
        actor_network.eval()
        print('actor loaded')


    t0 = time.time()
    is_done = False
    while not is_done:
        sys.stdout.flush(), sys.stderr.flush()
        obs = observation['observation']
        g = observation['desired_goal']
        print('obs: {}, g: {}'.format(obs.shape, g.shape))
        if load_ac:
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std)
            action = actor_network(inputs).detach().numpy().squeeze()
        else:
            action = env.action_space.sample()
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        print('t: {}, ag: {}, g: {}, reward: {}'.format(t, observation['achieved_goal'], observation['desired_goal'], reward))
    tf = time.time()
    print('Time simulated: {:.2f} seconds'.format(t/1000))
    print('Time taken: {:.2f} seconds'.format(tf-t0))


if __name__ == "__main__":
    main()
