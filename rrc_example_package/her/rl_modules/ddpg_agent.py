import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from rrc_example_package.her.mpi_utils.mpi_utils import sync_networks, sync_grads
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.rl_modules.models import actor, critic
from rrc_example_package.her.mpi_utils.normalizer import normalizer
from rrc_example_package.her.her_modules.her import her_sampler

import torch.nn as nn
from rrc_example_package.rearrange_dice_env import get_env_params
# import time

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    
    def __init__(self, args, env, env_params, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.args = args
        self.env = env
        self.env_params = get_env_params(self.env)
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)            
        # initialise related items
        self.init_items(self.actor_network, self.critic_network, self.env_params)
        self.model_path = os.path.join(self.args.save_dir, self.args.exp_dir)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        # Setup logs
        self.logs = {
                    'Epoch': [],
                    'ExploitSuccess': [],
                    'ExploreSuccess': [],
                    'LossPi': [],
                    'LossQ': []
                    }
        if self.args.load_pretrained:
            self.load_pretrained()
        
    def init_items(self, actor_network, critic_network, env_params, increment=False, update_wait=0):
        # collect data for update_wait epochs before updating model
        self.update_wait = update_wait
        # sync the networks across the cpus
        sync_networks(actor_network)
        sync_networks(critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(actor_network.state_dict())
        self.critic_target_network.load_state_dict(critic_network.state_dict())            
        # if use gpu
        if self.args.cuda:
            actor_network.cuda()
            critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward, self.args)
        # create the replay buffer
        self.buffer = replay_buffer(env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        if increment:
            # Only reinitialise o_norm, and load in previous mean/std for better initial exploration
            o_mean = self.o_norm.mean
            o_std = self.o_norm.std
            for i in range(increment):
                # assign existing dice norms to new dice
                # WARNING: adds too many in current implementation
                o_mean = np.concatenate((o_mean, o_mean[-3:]))
                o_std = np.concatenate((o_std, o_std[-3:]))
            # hacky solution: cut off extra
            o_mean = o_mean[:env_params['obs']]
            o_std = o_std[:env_params['obs']]
            self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
            self.o_norm.mean, self.o_norm.std = o_mean, o_std
        else:
            # create the normalizer
            self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
            self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)        

            
    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            self.epoch = epoch
            self.check_dice_increment()
            explore_success, loss_q, loss_pi = [], [], []
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        if self.epoch >= self.args.start_epochs:
                            action = self.get_action(obs, g)
                        else:
                            action = self.env.action_space.sample()
                        # feed the actions into the environment
                        observation, r, _, info = self.env.step(action)
                        obs_new = observation['observation']
                        ag_new = observation['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    explore_success.append(r)
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                if self.update_wait < 1:
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                    for _ in range(self.args.n_batches):
                        # train the network
                        q_loss, pi_loss = self._update_network()
                        loss_q.append(q_loss)
                        loss_pi.append(pi_loss)
                    # soft update
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
                else:
                    loss_q, loss_pi = 0, 0
            self.update_wait -= 1
            # Calc explore success rate
            explore_success = np.mean(explore_success)
            explore_success = MPI.COMM_WORLD.allreduce(explore_success, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            explore_success += 1
            # start to do the evaluation
            exploit_success = self._eval_agent()
            self.update_logs([epoch, exploit_success, explore_success, np.mean(loss_q), np.mean(loss_pi)])
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch: {} exploit: {:.3f} explore: {:.3f} loss_q: {:.3f} loss_pi: {:.3f}'.format(datetime.now(), epoch, exploit_success, explore_success, np.mean(loss_q), np.mean(loss_pi)))
                np.save(self.model_path + '/logs.npy', self.logs)
                if self.args.save_models and (epoch % self.args.save_freq) == 0:
                    # Save actor critic
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                                self.model_path + '/acmodel.pt')
    
    def update_logs(self, stats):
        for i, key in enumerate(self.logs.keys()):
            self.logs[key].append(stats[i])

    def get_action(self, obs, g, deterministic=False):
        g_norm = self.g_norm.normalize(g)
        obs_norm = self.o_norm.normalize(obs)
        # Choose action
        with torch.no_grad():
            inputs = np.concatenate([obs_norm, g_norm])
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            action = self.actor_network(inputs).cpu().numpy().squeeze()
            if not deterministic:
                action = self._select_actions(action)
        return action
        
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, action):
        # action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0] * mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        # obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        mb_obs, transitions['g'] = self._preproc_og(mb_obs, transitions['g'])
        # update
        self.o_norm.update(mb_obs)
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak_ddpg) * param.data + self.args.polyak_ddpg * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # do the normalization
        # concatenate the stuffs
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # start to do the update
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()
        
        return critic_loss.detach().numpy(), actor_loss.detach().numpy()
    
    
    def check_dice_increment(self): # TODO: rename
        def check_criteria_passed():
            return self.epoch % self.args.increment_freq == 0 and self.epoch > 0
        if self.args.increment_dice:
            if check_criteria_passed() and self.env.num_dice + self.args.increment_dice <= self.args.max_dice:
                # Increment number of dice in env
                self.env.set_num_dice(self.env.num_dice + self.args.increment_dice)
                self.env_params = get_env_params(self.env)
                extra_inputs = self.args.increment_dice * 3
                # Change ddpg items to match new env_params
                self.add_inputs(self.env_params, extra_inputs)
                
    def add_inputs(self, env_params, extra_inputs):
        # WARNING: replay buffer is completely emptied in this process
        self.actor_network.add_inputs(env_params, extra_inputs)
        self.critic_network.add_inputs(env_params, extra_inputs)
        self.init_items(self.actor_network, self.critic_network, env_params, increment=extra_inputs, update_wait=self.args.update_wait)
            
    def load_pretrained(self):
        model_path = 'pretrained/' + self.args.pretrained_dir
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] Loading in Actor Critic from {} ...'.format(datetime.now(), model_path))
        o_mean, o_std, g_mean, g_std, actor_state, critic_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network.load_state_dict(actor_state)
        self.critic_network.load_state_dict(critic_state)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # Load in pretrained norms
        self.o_norm.mean, self.o_norm.std = o_mean, o_std
        self.g_norm.mean, self.g_norm.std = g_mean, g_std
        # collect data for 5 epochs before updating model
        self.update_wait = self.args.update_wait
        self.args.start_epochs = 0

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                actions = self.get_action(obs, g, deterministic=True)
                observation, r, _, info = self.env.step(actions)
                obs = observation['observation']
                g = observation['desired_goal']
                per_success_rate.append(r)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return 1 + (global_success_rate / MPI.COMM_WORLD.Get_size())
