import numpy as np
import gym
import os, sys
from rrc_example_package.her.arguments import get_args
from mpi4py import MPI
from rrc_example_package.her.rl_modules.ddpg_agent import ddpg_agent
import random
import torch

from rrc_example_package import rearrange_dice_env
from rrc_example_package.her.rl_modules.sac import sac_agent
from rrc_example_package.benchmark_rrc.python.residual_learning.residual_wrappers import RandomizedEnvWrapper

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

def launch(args):
    if MPI.COMM_WORLD.Get_rank() == 0: print('\nArgs for {}:\n{}\n'.format(args.save_dir, args))
    
    # create the ddpg_agent
    if args.increment_dice:
        assert args.single_dice_focus and args.single_goal_focus, "Can only increment number of dice if using single__focus"
        
    env = rearrange_dice_env.SimtoRealRearrangeDiceEnv(
        step_size=args.step_size,
        sim=True,
        num_dice=args.num_dice,
        distance_threshold=args.distance_threshold,
        max_steps=args.max_steps,
        include_dice_velocity=args.include_dice_velocity,
        include_dice_orient=args.include_dice_orient,
        scale_dice=args.scale_dice,
        single_dice_focus=args.single_dice_focus,
        single_goal_focus=args.single_goal_focus,
        env_wrapped=(args.domain_randomization==1)
    )
    
    if args.domain_randomization == 1:
        if MPI.COMM_WORLD.Get_rank() == 0: print('Using Domain Randomization...')
        env = RandomizedEnvWrapper(env, flatten_obs=True, task='rearrange_dice')
        
    # set random seeds for reproduce
    seed = args.seed * MPI.COMM_WORLD.Get_rank()
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        
    # get the environment parameters
    env_params = rearrange_dice_env.get_env_params(env)
    
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     print('undo Q clip??')
    
    ddpg_trainer = ddpg_agent(args, env, env_params, seed=seed)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)