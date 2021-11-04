import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func, args, goal_len=3):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.seperate_her = args.seperate_her # TODO: rename as multi-criteria HER
        self.single_dice_focus = args.single_dice_focus
        self.single_goal_focus = args.single_goal_focus
        self.goal_len = goal_len # TODO: make global variable (or other)
        if self.single_dice_focus and not self.single_goal_focus:
            assert False

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        # TODO:
        # if not self.single_dice_focus:
        #    shuffle_dice_obs(transitions)
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        if self.seperate_her:
            # apply HER individually to each dice
            step = self.goal_len
        else:
            # apply HER across whole goal
            step = transitions['ag'].shape[1]
        # TODO: vectorize
        # Iterate through each 'sub-goal'/cube-goal and decide whther to apply HER
        for i in range(0, transitions['g'].shape[1], step):
            # her idx
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            # replace goal with achieved goal
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            if self.single_goal_focus and not self.single_dice_focus:
                chosen_dice = self.select_dice(episode_batch, episode_idxs, her_indexes)
                # prepare for slice
                dice_idx = chosen_dice * 3
                dice_idx = np.repeat(dice_idx, 3)
                dice_idx += np.tile(np.arange(3), chosen_dice.shape[0])
                trans_idx = np.arange(future_ag.shape[0])
                trans_idx = np.repeat(trans_idx, 3)
                # Only replace for selected dice
                transition_goals = transitions['g'][her_indexes]
                chosen_ag = future_ag[trans_idx, dice_idx].reshape((-1,3))
                transition_goals[:, i:i+step] = chosen_ag
                transitions['g'][her_indexes] = transition_goals
            else:
                # Only replace for selected dice
                transition_goals = transitions['g'][her_indexes]
                transition_goals[:, i:i+step] = future_ag[:, i:i+step]
                transitions['g'][her_indexes] = transition_goals
        if not self.single_goal_focus:
            # shuffle to ensure Q-function doesn't care about goal order
            transitions['g'] = self.shuffle_goal(transitions['g'])
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        
        return transitions
    
    def shuffle_goal(self, goals):
        num_dice = int(goals.shape[1]/self.goal_len)
        # TODO: vectorize + simplify
        for i in range(goals.shape[0]):
            goal_idxs = np.arange(num_dice)
            np.random.shuffle(goal_idxs)
            goal_idxs = np.repeat(goal_idxs, self.goal_len)
            goal_idxs *= 3
            add = np.arange(self.goal_len)
            add = np.tile(add, num_dice)
            goal_idxs += add
            goals[i] = goals[i, goal_idxs]
        return goals
    
    def select_dice(self, episode_batch, episode_idxs, her_indexes):
        # ATM, this function selects the dice which is closest to goal at end of episode
        final_ag = episode_batch['ag'][episode_idxs[her_indexes], -1]
        final_g = episode_batch['g'][episode_idxs[her_indexes], -1]
        distances = []
        for i in range(0, final_ag.shape[-1], 3):
            d = np.linalg.norm(final_ag[:,i:i+3] - final_g, axis=-1)
            distances.append(d)
        distances = np.array(distances)
        closest_dice = np.argmin(distances, axis=0)
        # closest_dice = np.zeros_like(closest_dice, dtype=int) # TODO: UNDO!!!
        return closest_dice
    