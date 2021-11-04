"""Example Gym environment for the RRC 2021 Phase 2."""
import enum
import pathlib
import typing

import gym
import numpy as np

import robot_fingers

import rrc_example_package.trifinger_simulation.python.trifinger_simulation as trifinger_simulation
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.tasks.rearrange_dice as task
from rrc_example_package.trifinger_simulation.python.trifinger_simulation import trifingerpro_limits
from rrc_example_package.trifinger_simulation.python.trifinger_simulation.camera import load_camera_parameters #, change_param_image_size
import rrc_example_package.trifinger_simulation.python.trifinger_simulation.visual_objects

from trifinger_cameras.utils import convert_image
from trifinger_object_tracking.py_lightblue_segmenter import segment_image

import cv2
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
import torch
from rrc_example_package.her.image2coords_sim import image2coords_sim
from rrc_example_package.her.image2coords_real import image2coords_real


CONFIG_DIR = pathlib.Path("/etc/trifingerpro")
if os.getcwd() == '/home/robert/summer_research/robochallenge/workspace': # TODO: Fix!
    SIM_CONFIG_DIR = pathlib.Path("src/rrc_example_package/camera_params")
else:
    SIM_CONFIG_DIR = pathlib.Path("camera_params")
SIM_CALIB_FILENAME_PATTERN = "camera{id}_cropped_and_downsampled.yml"

class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""
    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'ag': obs['achieved_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
        
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Env params: {}'.format(params))
    return params


class SimtoRealRearrangeDiceEnv(gym.GoalEnv):
    """Gym environment for rearranging dice with a TriFingerPro robot."""

    def __init__(
        self,
        provided_goal: typing.Optional[task.Goal] = None,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 1,
        sim: bool = True,
        visualization: bool = False,
        enable_cameras: bool = False,
        num_dice: int = 25,
        max_steps: int = 200,
        image_size=270,
        distance_threshold = 0.01,
        include_dice_velocity = False,
        include_dice_orient = False,
        scale_dice = 1,
        single_dice_focus = False,
        single_goal_focus = False,
        choose_strategy = False,
        env_wrapped = False,
        estimate_states = False,
        remote_run=False
    ):
        """Initialize.
        Args:
            goal: Goal pattern for the dice.  If ``None`` a new random goal is
                sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        # if provided_goal is not None:
        #     task.validate_goal(provided_goal)
        self.provided_goal = provided_goal
        self.action_type = action_type
        self.sim = sim
        self.visualization = visualization
        self.enable_cameras = enable_cameras
        self.set_num_dice(num_dice)
        self._max_episode_steps = max_steps
        self.image_size = image_size
        self.distance_threshold = distance_threshold
        self.include_dice_velocity = include_dice_velocity
        self.include_dice_orient = include_dice_orient
        self.single_dice_focus = single_dice_focus
        self.single_goal_focus = single_goal_focus
        self.choose_strategy = choose_strategy
        self.strategy_wait = 0
        self.env_wrapped = env_wrapped
        self.estimate_states = estimate_states
        task.EPISODE_LENGTH = max_steps * step_size
        self.og_width = task.DIE_WIDTH
        self.set_cube_scale(scale_dice)
        
        if sim:
            self.image2coords = image2coords_sim
        else:
            self.image2coords = image2coords_real
        
        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        else:
            self.step_size = step_size

        # will be initialized in reset()
        self.platform = None
        
        if self.enable_cameras:
            # load camera parameters
            if sim and not remote_run:
                param_dir = SIM_CONFIG_DIR
            else:
                param_dir = CONFIG_DIR
            self.camera_params = load_camera_parameters(
                param_dir, "camera{id}_cropped_and_downsampled.yml"
            )
            # self.camera_params = change_param_image_size(self.camera_params, image_size=image_size)
            

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )
        robot_tip_force_space = gym.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
        )

        mask_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 270, 270), dtype=np.uint8
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                        "tip_force": robot_tip_force_space,
                    } # TODO: add tip forces?
                ),
                "action": self.action_space,
                "desired_goal": mask_space,
                "achieved_goal": mask_space,
            }
        )

    def set_num_dice(self, num_dice):
        if num_dice < 1 or num_dice > 25:
            raise ValueError("num_dice must be > 0 and < 26.")
        else:
            self.num_dice = num_dice
            if self.sim:
                task.NUM_DICE = num_dice
            
            
    def set_cube_scale(self, scale):
        self.cube_scale = scale

    def compute_reward(
        self,
        achieved_goal: typing.Sequence[np.ndarray],
        desired_goal: typing.Sequence[np.ndarray],
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.
        Args:
            achieved_goal: Segmentation mask of the observed camera images.
            desired_goal: Segmentation mask of the goal positions.
            info: Unused.
        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        # if self.enable_cameras:
        #     return -task.evaluate_state(desired_goal, achieved_goal)
        # else:
        return self.sparse_reward(desired_goal, achieved_goal)
        
    def sparse_reward(self, desired_goal, achieved_goal):
        """
        If a goal has been achieved (a dice is within threshold), its reward is 0. Else it is -1
        """
        g_dim = 3
        dice_in_goal = int(desired_goal.shape[-1] / g_dim)
        dice_in_ag = int(achieved_goal.shape[-1] / g_dim)
        r_shape = list(desired_goal.shape)
        r_shape[-1] = dice_in_goal
        reward = np.zeros(tuple(r_shape))
        if self.single_goal_focus:
            assert dice_in_goal == 1
        else:
            assert self.num_dice == int(desired_goal.shape[-1] / g_dim)
        if self.single_dice_focus:
            assert dice_in_ag == 1
        # assert desired_goal.shape == achieved_goal.shape
        for g in range(dice_in_goal):
            g_idx = g * g_dim
            check_g = desired_goal[..., g_idx:g_idx+g_dim]
            for ag in range(dice_in_ag):
                ag_idx = ag * g_dim
                check_ag = achieved_goal[..., ag_idx:ag_idx+g_dim]
                d = np.linalg.norm(check_ag - check_g, axis=-1)
                reward[..., g] += d < self.distance_threshold
        reward = reward <= 0
        # Mean ensures reward at each step is between 0 and -1
        reward = -np.mean(reward, axis=-1)
        return reward
    
    def check_dice_at_goal(self, dice_pos, goals):
        assert len(dice_pos) == len(goals)
        distances = np.zeros((len(goals), len(dice_pos)))
        for i, goal in enumerate(goals):
            for j, die in enumerate(dice_pos):
                d = np.linalg.norm(np.array(die) - np.array(goal))
                distances[i,j] = d
        dice_at_goal = distances < self.distance_threshold
        dice_at_goal = np.sum(dice_at_goal, axis=0)
        dice_at_goal = dice_at_goal > 0
        return dice_at_goal
    
    def choose_diceNgoal(self, dice_pos, goals, pick_furthest=False):
        if self.strategy_wait > 0:
            self.strategy_wait -= 1
        else:
            distances = np.zeros((len(goals), len(dice_pos)))
            for i, goal in enumerate(goals):
                for j, die in enumerate(dice_pos):
                    d = np.linalg.norm(np.array(die) - np.array(goal))
                    distances[i,j] = d
            distance_complete = distances < self.distance_threshold
            times_goal_complete = np.sum(distance_complete, axis=1)
            nearest_goals = []
            dist_to_nearest_goals = []
            for j, die in enumerate(dice_pos):
                dist_to_goals = distances[:,j]
                dist_to_goals += ((times_goal_complete - distance_complete[:,j]) > 0) * 1e6
                nearest_goal = np.argmin(dist_to_goals)
                dist_to_nearest_goal = np.amin(dist_to_goals)
                nearest_goals.append(nearest_goal)
                dist_to_nearest_goals.append(dist_to_nearest_goal)
            dist_to_nearest_goals = np.array(dist_to_nearest_goals)
            dist_to_nearest_goals += (dist_to_nearest_goals < self.distance_threshold) * 1e6
            chosen_dice = np.argmin(dist_to_nearest_goals)
            chosen_goal = nearest_goals[chosen_dice]
            self.chosen_dice, self.chosen_goal = chosen_dice, chosen_goal
            self.strategy_wait = 15
        # swap dice
        if self.chosen_dice >= len(dice_pos):
            dice_idx = 0
        else:
            dice_idx = self.chosen_dice
        replaced_dice = dice_pos[0]
        dice_pos[0] = dice_pos[dice_idx]
        dice_pos[dice_idx] = replaced_dice
        # swap goal
        replaced_goal = goals[0]
        goals[0] = goals[self.chosen_goal]
        goals[self.chosen_goal] = replaced_goal
        
        return dice_pos, goals
        
                        
    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.
        .. note::
           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.
        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]
    
    def estimate_states_from_image(self, camera_observation):
        assert self.enable_cameras, "Can only estimate states from images if cameras are enabled!"
        cube_pos = self.image2coords(camera_observation, self.camera_params)
        return cube_pos, None, None, None
    
    def select_closest2dice1(self, dice_pos):
        if len(dice_pos) > self.num_dice:
            dice1 = dice_pos[0]
            dist_to_dice1 = [np.linalg.norm(np.array(dice1) - np.array(other_dice)) for other_dice in dice_pos[1:]]
            new_dice_pos = [dice1]
            while len(new_dice_pos) < self.num_dice:
                min_dist = min(dist_to_dice1)
                min_idx = dist_to_dice1.index(min_dist)
                new_dice_pos.append(dice_pos[min_idx+1])
                dist_to_dice1[min_idx] += 1e6
        if len(dice_pos) < self.num_dice:
            new_dice_pos = dice_pos
            i = 0
            while len(new_dice_pos) < self.num_dice:
                new_dice_pos.append(dice_pos[i])
                i += 1
        return new_dice_pos

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        
        # Get masks
        if self.enable_cameras:
            if self.sim:
                segmentation_masks = [
                    segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
                    for c in camera_observation.cameras
                ]
            else:
                segmentation_masks = [
                    segment_image(convert_image(c.image))
                    for c in camera_observation.cameras
                ]
        else:
            segmentation_masks = None
        
        # Get states
        if self.sim and not self.estimate_states:
            dice_pos, dice_orient, dice_lin_vel, dice_ang_vel = self.platform.get_dice_states()
        else:
            dice_pos, dice_orient, dice_lin_vel, dice_ang_vel = self.estimate_states_from_image(camera_observation)
        
        goal = self.goal.copy()
        if self.choose_strategy:
            assert self.single_dice_focus and self.single_goal_focus
            dice_pos, goal = self.choose_diceNgoal(dice_pos, goal)
            
        if len(dice_pos) != self.num_dice:
            dice_pos = self.select_closest2dice1(dice_pos)
            
        observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
                "tip_force": robot_observation.tip_force
            },
            "object_observation": {
                "position": dice_pos,
                "orientation": dice_orient,
                "linear_velocity": dice_lin_vel,
                "angular_velocity": dice_ang_vel
            },
            "action": action,
            "desired_goal": goal, # TODO: remove redundant z dimension?? - but beware effects on reward and HER calcs
            "achieved_goal": dice_pos,
            "image_masks": segmentation_masks
        }
        
        if self.single_dice_focus:
            observation["achieved_goal"] = observation["achieved_goal"][0]
        if self.single_goal_focus:
            observation["desired_goal"] = observation["desired_goal"][0]
                
        return observation
    
    def flatten_obs(self, obs):
        # TODO: include tipforces?, remove torque?, include previous action?
        state_obs = obs["robot_observation"]["position"]
        state_obs = np.concatenate((state_obs, obs['robot_observation']['velocity']))
        state_obs = np.concatenate((state_obs, obs['robot_observation']['torque']))
        # state_obs = np.concatenate((state_obs, obs['robot_observation']['tip_force']))
        # # If using states rather than images
        # if not self.enable_cameras:
            
        self.obs_dice_start_idx = state_obs.shape[0]
        positions = np.array(obs["object_observation"]['position']).flatten()
        state_obs = np.concatenate((state_obs, positions))
        
        # WARNING: Not accounting for camera update delays!!!
        if self.include_dice_velocity:
            assert self.sim, "Dice velocities not implemented for real robot"
            lin_vels = np.array(obs["object_observation"]['linear_velocity']).flatten()
            state_obs = np.concatenate((state_obs, lin_vels))
            
        if self.include_dice_orient:
            assert self.sim, "Dice orientations not implemented for real robot"
            orientations = np.array(obs["object_observation"]['orientation']).flatten()
            state_obs = np.concatenate((state_obs, orientations))
            if self.include_dice_velocity:
                ang_vels = np.array(obs["object_observation"]['angular_velocity']).flatten()
                state_obs = np.concatenate((state_obs, ang_vels))
        
        flat_obs = {
            "observation": state_obs,
            "desired_goal": np.array(obs["desired_goal"]).flatten(),
            "achieved_goal": np.array(obs["achieved_goal"]).flatten(),
            }
        
        if self.enable_cameras:
            flat_obs['image_masks'] = obs['image_masks']
        
        return flat_obs
    
    def _update_obj_vel(self, observation, initial):
        # TODO: implement if not using sim velocities
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action, initial=False):
        """Run one timestep of the environment's dynamics.
        Important: ``reset()`` needs to be called before doing the first step.
        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).
        Returns:
            tuple:
            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            print('Action space: {}'.format(self.action_space))
            print('Input action: {}'.format(action))
            raise ValueError(
                "Given action is not contained in the action space."
            )
        
        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)
            
        # TODO: fix actions when domain randomizing? (Don't reveal true action to agent in obs?)
        reward = 0.0
        robot_action = self._gym_action_to_robot_action(action)
        for _ in range(num_steps):
            # send action to robot
            t = self.platform.append_desired_action(robot_action)
            # make sure to not exceed the episode length
            if initial or t >= task.EPISODE_LENGTH - 1:
                break
            
        self.info["time_index"] = t    
        observation = self._create_observation(t, action)
        # Flatten and update obs here if env is not wrapped, else do it in wrapper env
        if not self.env_wrapped:
            observation = self.flatten_obs(observation)
        
        reward += self.compute_reward(
            np.array(observation["achieved_goal"]).flatten(),
            np.array(observation["desired_goal"]).flatten(),
            self.info,
        )
        
        is_done = t >= task.EPISODE_LENGTH
        
        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if not self.sim and self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )
        
        if self.sim:
            task.DIE_WIDTH = self.og_width * self.cube_scale
            
            # hard-reset simulation
            del self.platform
            
            self.platform = trifinger_simulation.TriFingerPlatform(
                visualization=self.visualization,
                object_type=trifinger_simulation.trifinger_platform.ObjectType.DICE,
                enable_cameras=self.enable_cameras,
                config_dir=SIM_CONFIG_DIR,
                calib_filename_pattern=SIM_CALIB_FILENAME_PATTERN
            )
        else:
            self.platform = robot_fingers.TriFingerPlatformFrontend()

        # if no goal is given, sample one randomly
        if self.provided_goal is None:
            self.goal = task.sample_goal()
        else:
            self.goal = self.provided_goal
        
        if self.enable_cameras:
            self.goal_masks = task.generate_goal_mask(self.camera_params, self.goal) # TODO: visualize
            
        # visualize the goal
        if self.visualization and self.sim:
            self.goal_markers = []
            for g in self.goal:
                goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                    width=task.DIE_WIDTH,
                    position=g,
                    orientation=(0, 0, 0, 1),
                    pybullet_client_id=self.platform.simfinger._pybullet_client_id,
                )
                self.goal_markers.append(goal_marker)

        self.info = {"time_index": -1}

        # need to already do one step to get initial observation
        observation, _, _, _ = self.step(self._initial_action, initial=True)

        return observation



class RealRobotRearrangeDiceEnv(gym.GoalEnv):
    """Gym environment for rearranging dice with a TriFingerPro robot."""

    def __init__(
        self,
        goal: typing.Optional[task.Goal] = None,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 1,
    ):
        """Initialize.
        Args:
            goal: Goal pattern for the dice.  If ``None`` a new random goal is
                sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal is not None:
            task.validate_goal(goal)
        self.goal = goal

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # load camera parameters
        self.camera_params = load_camera_parameters(
            CONFIG_DIR, "camera{id}_cropped_and_downsampled.yml"
        )

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        mask_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 270, 270), dtype=np.uint8
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "action": self.action_space,
                "desired_goal": mask_space,
                "achieved_goal": mask_space,
            }
        )

    def compute_reward(
        self,
        achieved_goal: typing.Sequence[np.ndarray],
        desired_goal: typing.Sequence[np.ndarray],
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.
        Args:
            achieved_goal: Segmentation mask of the observed camera images.
            desired_goal: Segmentation mask of the goal positions.
            info: Unused.
        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        return -task.evaluate_state(desired_goal, achieved_goal)

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.
        .. note::
           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.
        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)

        segmentation_masks = [
            segment_image(convert_image(c.image))
            for c in camera_observation.cameras
        ]

        observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "action": action,
            "desired_goal": self.goal_masks,
            "achieved_goal": segmentation_masks,
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Important: ``reset()`` needs to be called before doing the first step.
        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).
        Returns:
            tuple:
            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

            observation = self._create_observation(t, action)

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            # make sure to not exceed the episode length
            if t >= task.EPISODE_LENGTH - 1:
                break

        is_done = t >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            goal = task.sample_goal()
        else:
            goal = self.goal

        self.goal_masks = task.generate_goal_mask(self.camera_params, goal)

        self.info = {"time_index": -1}

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation