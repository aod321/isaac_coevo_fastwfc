from abc import ABC, abstractmethod
from collections import deque
import queue
import numpy as np
import random
from threading import local
import time
import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
#from isaacgym import gymtorch
from math import sqrt
import math
# from omegaconf import read_write
import torch
import cv2
import gym
from gym import spaces
import copy
from stable_baselines3.common.vec_env import VecEnvWrapper
from tqdm.rich import tqdm
import os

from draw import *

import fastwfc

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv

N_DISCRETE_ACTIONS = 6
N_CHANNELS = 3
HEIGHT = 84
WIDTH = 84


class Wave(object):
    def __init__(self, wave) -> None:
        self.wave = wave
        self.seed = np.array(self.wave).astype(np.int32)
        # 从1开始
        self.seed[:, 0]+=1
        self.seed = self.seed.reshape(-1,1,2)
    
class Block(object):
    def __init__(self, handler, used, type, tag):
        self.handler = handler
        self.used = used
        self.type = type
        self.tag = tag

    def set_handler(self, handler, type, tag):
        self.handler = handler
        self.tyep = type
        self.used = False
        self.tag = tag

    def is_used(self):
        self.used = True
    
    def __str__(self) -> str:
        return self.tag


class VecAdapter(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv=venv)
        self.done = None
        self.t_start = 0

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions 

    def reset(self):
        return self.venv.reset()

    def seed(self, seed) -> None:
        pass

    def step_wait(self):
        return self.venv.step(self.actions)


class StableBaselinesVecEnvAdapter(VecEnv):

    def step_async(self, actions):
        pass

    def step_wait(self):
        obs, rewards, dones, info_dict = self.venv.step(self.actions)
        return obs["obs"], rewards, dones, info_dict

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, seed):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            n = self.num_envs
        else:
            n = len(indices)
        return [False] * n


class PCGVecEnv(StableBaselinesVecEnvAdapter):

    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    # Set this in SOME subclasses
    metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, num_envs=16, observation_space=None, action_space=None, headless_: bool = True, render_indicator: bool = True,
                compute_device_id = 0, graphics_device_id = 0,prefab_size=2, prefab_height=2, height_scale=0.7):
        # Define action and observation space
        if observation_space is None:
            observation_space = spaces.Box(low=0, high=255, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        if action_space is None:
            action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self._action_space = action_space
        self._obs_space = observation_space
        self.headless = headless_
        super(PCGVecEnv, self).__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)
        self.observation_space = observation_space
        self.action_space = action_space
        # torch.manual_seed(3407)
        self.prefab_size = prefab_size
        self.height_scale = height_scale
        self.prefab_height = prefab_height * height_scale
        self.timer0 = time.time()
        '''
        simulatiom parameters
        '''
        self.num_matrix_envs = 16
        self.headless = headless_
        self.spd = 10

        self.render_indicator = render_indicator

        self.initial_height = 2.05

        gravity_scale = 4.5

        # get default set of parameters
        sim_params = gymapi.SimParams()
        # args = gymutil.parse_arguments()
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 2
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        sim_params.substeps = 2
        sim_params.dt = 1.0 / 30.0
        # setting up the Z-up axis
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8 * gravity_scale)

        # sim_params.physx.max_gpu_contact_pairs = sim_params.physx.max_gpu_contact_pairs*20

        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 1
        plane_params.static_friction = 1 / gravity_scale
        plane_params.dynamic_friction = 1 / gravity_scale
        plane_params.restitution = 0.1

        # set up the env grid
        spacing = 20
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Attractor setup
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e10
        attractor_properties.damping = 0
        attractor_properties.axes = gymapi.AXIS_ROTATION
        attractor_pose = gymapi.Transform()
        attractor_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        attractor_pose.r = gymapi.Quat(0,1,0,1)

        # create procedural asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        asset_options.fix_base_link = False
        asset_options.linear_damping = 0.1
        asset_options1 = gymapi.AssetOptions()
        asset_options1.collapse_fixed_joints = True
        asset_options1.enable_gyroscopic_forces = False
        asset_options1.density = 0.02
        asset_options1.fix_base_link = True
        asset_options1.linear_damping = 0.1
        asset_options1.disable_gravity = True

        color_red = gymapi.Vec3(1,0,0)
        color_green = gymapi.Vec3(0,1,0)

        # camera sensor properties
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.width = WIDTH
        self.camera_properties.height = HEIGHT
        self.camera_properties.enable_tensors = True

        # rigid shape material properties
        self.shape_props = gymapi.RigidShapeProperties()
        self.shape_props.friction = -1 / gravity_scale
        self.shape_props.rolling_friction = 1 / gravity_scale
        self.shape_props.torsion_friction = 1 / gravity_scale
        self.shape_props.compliance = 0
        self.shape_props.restitution = 0
        self.shape_props1 = gymapi.RigidShapeProperties()
        self.shape_props1.friction = 0.2 / gravity_scale
        self.shape_props1.rolling_friction = -1 / gravity_scale
        self.shape_props1.torsion_friction = -1 / gravity_scale
        self.shape_props1.compliance = 0
        self.shape_props1.restitution = 0
        self.shape_props2 = gymapi.RigidShapeProperties()
        self.shape_props2.friction = 1 / gravity_scale
        self.shape_props2.rolling_friction = -1 / gravity_scale
        self.shape_props2.torsion_friction = -1 / gravity_scale
        self.shape_props2.compliance = 0
        self.shape_props2.restitution = 0

        shape_props = self.shape_props
        shape_props1 = self.shape_props1
        shape_props2 = self.shape_props2

        # single world camera image
        self.camera_sensor_image = np.zeros((HEIGHT,WIDTH, 3), np.uint8)
        # human control
        self.interaction = 0

        # proximity threshold for reward calculation
        self.proximity_threshold = 1.5
        self.max_steps = 1000

        self.show_action = False

        self.all_rews = np.zeros(self.num_matrix_envs, dtype=np.float32)
        self.all_dones = np.zeros(self.num_matrix_envs, dtype=np.bool8)
        self.vec_obs = np.zeros((self.num_matrix_envs, N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        self.all_rews_backup = self.all_rews.copy()
        self.all_dones_backup = self.all_dones.copy()

        # reward fifo of max length 100
        self.reward_fifo = deque(maxlen=300)
        self.best_performance = 0
        self.reward_archive = []
        self.rew_avg = []
        for i in range(self.num_matrix_envs):
            self.reward_archive.append(deque(maxlen=100))
            self.rew_avg.append(0)

        self.obs_buffer4 = np.zeros((HEIGHT,WIDTH, 4), np.uint8)
        self.obs_buffer3 = np.zeros((HEIGHT,WIDTH, 3), np.uint8)
        # self.obs_buffer3t = np.zeros((HEIGHT,WIDTH, 3), np.uint8)

        '''
        generate WFC maps
        '''

        self.wfcworker_ = fastwfc.XLandWFC("samples.xml")
        wave = self.wfcworker_.get_ids_from_wave(self.wfcworker_.build_a_open_area_wave())
        wave = Wave(wave)
        
        # WFC map workspace
        self.seeds = [wave]
        self.seeds_collection = deque(maxlen=64)
        self.space = self.get_space_from_wave()

        for i in range(0,self.num_matrix_envs-1):
            self.append_seed(wave)

        # print("seeds generated : ",type(seeds))

        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        print("graphics_device_id : ",compute_device_id)
        print("compute_device_id : ",graphics_device_id)

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

        # load PCG asset
        asset_root = "./assets"
        color_list =  ["gray", "blue", "yellow", "orange", "red", "white"]
        self.color_name_list = color_list
        all_pt = {}
        # for example: pt_cubes = ["PCG/gray_cube.urdf","PCG/blue_cube.urdf","PCG/yellow_cube.urdf", "PCG/orange_cube.urdf", "PCG/red_cube.urdf", "PCG/white_cube.urdf"]
        for name in ["cube", "ramp", "corner"]:
            all_pt[name] = [f"PCG/{c}_{name}.urdf" for c in color_list]
        all_assets = {}
        for name in ["cube", "ramp", "corner"]:
            all_assets[name] = [self.gym.load_asset(self.sim, asset_root, pt, asset_options1) for pt in all_pt[name]]
        capsule_asset = self.gym.create_capsule(self.sim, 1, 1, asset_options)
        assets_dict = {
        "cube_assets": all_assets["cube"],
        "ramp_assets": all_assets["ramp"],
        "corner_assets": all_assets["corner"],
        "capsule_asset": capsule_asset
        }
        self.capsule_asset =  assets_dict["capsule_asset"]
        self.cube_assets = assets_dict["cube_assets"]
        self.ramp_assets = assets_dict["ramp_assets"]
        self.corner_assets = assets_dict["corner_assets"]

        if not self.headless:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)

        self.envs = []
        self.camera_handles = []
        self.actor_handles = []
        self.food_handles = []
        self.attractor_handles = []
        self.all_cubes_handler = {}
        self.all_ramps_handler = {}
        self.all_corners_handler = {}
        self.grid_tile_blocks = {}
        self.initial_pos = gymapi.Transform()
        self.initial_pos.p = gymapi.Vec3(0, 0, 0)
        self.initial_pos.r = gymapi.Quat(0,0,0,1)

        self.actor_scales = []
        self.cube_capacity_per_level = 9*9
        self.corner_capacity_per_level = 9*3        # to be tested
        self.ramp_capacity_per_level = 9*3

        self.vanish_scale = 0.000001

        '''
        1. build all actors
        '''
        for n in range(self.num_matrix_envs):
            self.envs.append(self.gym.create_env(self.sim, env_lower, env_upper, 4))
            self.__create_all_actors(n)
        '''
        2. arrange blocks in the environment
        '''
        for n in range(self.num_matrix_envs):
            self.__gridRender(self.seeds[n], n)
            self.placeAgentAndFood(n)
        
        env_actor_count = self.gym.get_actor_count(self.envs[0])
        print("env_actor_count: ", env_actor_count)

        # a must-have operation 
        self.gym.prepare_sim(self.sim)

        # step count for each environment
        self._step_counts = np.zeros(self.num_matrix_envs)
        self._step_counts_backup = self._step_counts.copy()
        # Used for logging/debugging
        self._episode_rewards = np.zeros(self.num_matrix_envs)

        if not self.headless:
            cam_pos = gymapi.Vec3(0.0, -15.0, 15.0)
            cam_target = gymapi.Vec3(4, 4, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # time test
        self.start_time = time.time()
        self.last_frame_cnt = 0
        self.step_count = 0

        # Create helper geometry used for visualization
        # Create an wireframe axis
        self.axes_geom = gymutil.AxesGeometry(2)

        self.facings = []
        for i in range(self.num_matrix_envs):
            self.facings.append(0.0)
        self.facing_step = 0.25
        self.facings_backup = copy.deepcopy(self.facings)

        # local coordinate system
        self.verts = np.empty((3, 1), gymapi.Vec3.dtype)
        self.verts[0][0] = (1, 0, 0)
        self.verts[1][0] = (0, 1, 0)
        self.verts[2][0] = (0, 0, 1)

        # store initial status for all envs
        self.initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

        self.sim_status = []
        for i in range(self.num_matrix_envs):
            env_status = np.copy(self.gym.get_env_rigid_body_states(self.envs[i], gymapi.STATE_ALL))
            self.sim_status.append(env_status)

    def placeAgentAndFood(self, env_id):
        space = self.space.copy()
        assert len(space) > 1, len(space)
        self.agent_space = list(space)
        # choose a place for agent
        random_cell = np.random.choice(self.agent_space)
        cell_top_tile_handler = self.grid_tile_blocks[env_id][random_cell][-1].handler
        # reset agent's pose
        cell_body_states = self.gym.get_actor_rigid_body_states(self.envs[env_id], cell_top_tile_handler, gymapi.STATE_ALL)
        actor_cell_pose = gymapi.Transform()
        actor_cell_pose.p = gymapi.Vec3(cell_body_states["pose"]["p"][0][0], cell_body_states["pose"]["p"][0][1], cell_body_states["pose"]["p"][0][2] + self.prefab_height * 1.5)
        actor_cell_pose.r = gymapi.Quat(0, 1, 0, 1)
        # choose a place for food
        self.food_space = self.agent_space.copy()
        self.food_space.remove(random_cell)
        random_cell = np.random.choice(list(self.food_space))
        cell_top_tile_handler = self.grid_tile_blocks[env_id][random_cell][-1].handler
        cell_body_states = self.gym.get_actor_rigid_body_states(self.envs[env_id], cell_top_tile_handler, gymapi.STATE_ALL)
        food_cell_pose = gymapi.Transform()
        food_cell_pose.p = gymapi.Vec3(cell_body_states["pose"]["p"][0][0], cell_body_states["pose"]["p"][0][1], cell_body_states["pose"]["p"][0][2] + self.prefab_height * 1.5)
        food_cell_pose.r = gymapi.Quat(0, 1, 0, 1)
        # reset agent's pose
        self.__moveActor(env_id, self.actor_handles[env_id], actor_cell_pose)
        # reset food's pose
        self.__moveActor(env_id, self.food_handles[env_id], food_cell_pose)
        
    def get_space_from_wave(self):
        return list(np.arange(81).astype(np.int32))

    def __instanteAsset(self, env_id, asset, pos, name, colli_group, colli_filter):
        return self.gym.create_actor(self.envs[env_id], asset, pos, name, colli_group, colli_filter)
    
    def __create_all_actors(self, env_id):
        # create all cubes, and store them regularly
        self.all_cubes_handler[env_id] = {c: []for c in self.color_name_list}
        for i,color in enumerate(self.color_name_list):
            cube_asset = self.cube_assets[i]
            for j in range(self.cube_capacity_per_level):
                handler = self.__instanteAsset(env_id=env_id,asset=cube_asset, pos=self.initial_pos, name=f"{color}_cube_{j}",
                colli_group=env_id, colli_filter=0)
                self.gym.set_actor_scale(self.envs[env_id], handler, self.vanish_scale)
                self.all_cubes_handler[env_id][color].append(Block(handler=handler, used=False, type="cube", tag=f"{color}_cube_{j}"))
        
        # create all ramps, and store them regularly
        self.all_ramps_handler[env_id] = {c: []for c in self.color_name_list}
        for i,color in enumerate(self.color_name_list):
            if color == "gray":
                self.all_ramps_handler[env_id][color] = []
                continue
            ramp_asset = self.ramp_assets[i]
            for j in range(self.ramp_capacity_per_level):
                handler = self.__instanteAsset(env_id=env_id,asset=ramp_asset, pos=self.initial_pos, name=f"{color}_ramp_{j}",
                colli_group=env_id, colli_filter=0)
                self.gym.set_actor_scale(self.envs[env_id], handler, self.vanish_scale)
                self.all_ramps_handler[env_id][color].append(Block(handler=handler, used=False, type="ramp", tag=f"{color}_ramp__{j}"))
        # create all corners, and store them regularly
        self.all_corners_handler[env_id] = {c: []for c in self.color_name_list}
        for i,color in enumerate(self.color_name_list):
            if color == "gray":
                self.all_ramps_handler[env_id][color] = []
                continue
            corner_asset = self.corner_assets[i]
            for j in range(self.corner_capacity_per_level):
                handler = self.__instanteAsset(env_id=env_id,asset=corner_asset, pos=self.initial_pos, name=f"{color}_corner_{j}",
                colli_group=env_id, colli_filter=0)
                self.gym.set_actor_scale(self.envs[env_id], handler, self.vanish_scale)
                self.all_corners_handler[env_id][color].append(Block(handler=handler, used=False, type="corner", tag=f"{color}_corner_{j}"))
        self.__createActorAndFood(env_id)

    def __createActorAndFood(self, env_id):
            # create capsule_asset actor in the environment
        color_red = gymapi.Vec3(1,0,0)
        color_green = gymapi.Vec3(0,1,0)
        # generate random x,y coordinates in range [0,18] for the actor
        x = random.uniform(0,16)
        y = random.uniform(0,16)
        facing_ = 0.25 * (float)(random.randint(0,8))
        r_ = gymapi.Quat(0,1,0,1) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), facing_*math.pi)
        # r_ = gymapi.Quat(0,1,0,1)
        startpose = gymapi.Transform()
        startpose.p = gymapi.Vec3(x, y, self.initial_height)
        startpose.r = gymapi.Quat(0,1,0,1)
        cap_handle = self.gym.create_actor(self.envs[env_id], self.capsule_asset, startpose, 'agent', env_id, 0)
        self.gym.set_actor_rigid_shape_properties(self.envs[env_id], cap_handle, [self.shape_props2])
        self.gym.set_actor_scale(self.envs[env_id], cap_handle, 0.4)
        self.gym.set_rigid_body_color(self.envs[env_id], cap_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_red)
        self.actor_handles.append(cap_handle)
        # set random position for food
        x = random.uniform(0,16)
        y = random.uniform(0,16)
        # create capsule_asset food in the environment
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(x, y, self.initial_height)
        pose.r = gymapi.Quat(0,1,0,1)
        food_handle = self.gym.create_actor(self.envs[env_id], self.capsule_asset, pose, 'food', env_id, 0)
        self.gym.set_actor_rigid_shape_properties(self.envs[env_id], food_handle, [self.shape_props2])
        self.gym.set_actor_scale(self.envs[env_id], food_handle, 0.6)
        # self.food_handles.append(food_handle)
        self.gym.set_rigid_body_color(self.envs[env_id], food_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_green)
        self.food_handler = food_handle
        self.food_handles.append(food_handle)
        # keep agent and food stand
        att_pose = gymapi.Transform()
        att_pose.p = gymapi.Vec3(x,y, self.initial_height)
        att_pose.r = r_
        attractor_properties_ = gymapi.AttractorProperties()
        attractor_properties_.stiffness = 5e10
        attractor_properties_.damping = 0
        # attractor_properties_.axes = gymapi.AXIS_SWING_1 | gymapi.AXIS_SWING_2  # 48
        attractor_properties_.axes = gymapi.AXIS_ROTATION
        # attractor_properties_.axes = 0
        attractor_properties_.target = att_pose
        attractor_properties_.rigid_handle = cap_handle
        attractor_handle_ = self.gym.create_rigid_body_attractor(self.envs[env_id], attractor_properties_)
        self.attractor_handles.append(attractor_handle_)
                
        attractor_properties_food = gymapi.AttractorProperties()
        attractor_properties_food.stiffness = 5e10
        attractor_properties_food.damping = 0
        # attractor_properties_food.axes = gymapi.AXIS_SWING_1 | gymapi.AXIS_SWING_2
        attractor_properties_food.axes = gymapi.AXIS_ROTATION
        attractor_properties_food.target = pose
        attractor_properties_food.rigid_handle = food_handle
        attractor_handle_food = self.gym.create_rigid_body_attractor(self.envs[env_id], attractor_properties_food)

        h1 = self.gym.create_camera_sensor(self.envs[env_id], self.camera_properties)
        # camera_offset = gymapi.Vec3(-2, 0, 0)
        camera_offset = gymapi.Vec3(-2, 0, 0)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.4*math.pi)
        # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        # camera_rotation = gymapi.Quat(0,,0,1)
        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_id], cap_handle, 0)
        self.gym.attach_camera_to_body(h1, self.envs[env_id], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
        self.camera_handles.append(h1)
        # self.agent_idx = self.gym.get_actor_index(self.envs[env_id], self.agent_handler, gymapi.DOMAIN_SIM)
        # self.food_idx = self.gym.get_actor_index(self.envs[env_id], self.food_handler, gymapi.DOMAIN_SIM)

    @property
    def step_counts(self):
        return self._step_counts.copy()
    
    @property
    def episode_rewards(self):
        return self._episode_rewards.copy()
    
    @property
    def auto_reset_after_done(self):
        return self._auto_reset_after_done

    def append_seed(self, seed):
        self.seeds.append(seed)
        # 2. add new collection
        landscape_new = True
        for s in self.seeds_collection:
            if s == seed:
                landscape_new = False
                break
        if landscape_new:
            self.seeds_collection.append(copy.deepcopy(seed))

    def in_collection(self,seed):
        for s in self.seeds_collection:
            if s == seed:
                return True
        return False

    def generate_decendent(self):
        base_wave = self.seeds_collection[-1].wave
        new_wave, _ = self.wfcworker_.mutate(base_wave=self.wfcworker_.wave_from_id(base_wave), new_weight=162, iter_count=1, out_img=False)
        new_seed = Wave(new_wave)
        while new_wave == base_wave or self.in_collection(new_seed)==True:
            new_wave, _ = self.wfcworker_.mutate(base_wave=self.wfcworker_.wave_from_id(base_wave), new_weight=162, iter_count=1, out_img=False)
        new_seed = Wave(new_wave)
        # update collection
        self.seeds_collection.append(new_seed)
        return copy.deepcopy(new_seed)

    # used for ray parrallel processing
    def accept_decendent(self, new_seed):
        # update collection
        self.seeds_collection.append(new_seed)

    def update_collection(self, seeds_collection):
        
        self.seeds_collection = copy.deepcopy(seeds_collection)

    def seed_distribution(self):
        
        distribution_table = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2],
            [0,0,0,1,1,1,2,2,2,2,2,3,3,3,3,3],
            [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4],
            [0,0,1,1,2,2,3,3,3,4,4,4,5,5,5,5],
            [0,1,1,2,2,3,3,4,4,5,5,5,6,6,6,6],
            [0,1,2,2,3,3,4,4,5,5,6,6,7,7,7,7],
            [0,1,2,3,4,4,5,5,6,6,7,7,8,8,8,8],
            [0,1,2,3,4,5,6,6,7,7,8,8,9,9,9,9],
            [0,1,2,3,4,5,6,7,8,8,9,9,10,10,10,10],
            [0,1,2,3,4,5,6,7,8,9,10,10,11,11,11,11],
            [0,1,2,3,4,5,6,7,8,9,10,11,12,12,12,12]
        ]
        collection_size = len(self.seeds_collection)

        if collection_size < 13:
            distribution = copy.deepcopy(distribution_table[collection_size-1])
            arrangement = []
            for i in distribution:
                arrangement.append(-(collection_size-i))
        else:
            distribution = copy.deepcopy(distribution_table[12])
            arrangement = []
            for i in distribution:
                arrangement.append((collection_size-13)-(collection_size-i))
        collection_ = []
        for i in arrangement:
            collection_.append(copy.deepcopy(self.seeds_collection[i]))

        # alter landscape
        for i in range(len(collection_)):
            self.set_landscape(i, collection_[i], update_collection = False)

    def revert_map_collection(self):
        # 
        self.seeds_collection.pop()

    # site reservation
    def pause(self):
        # save physical states
        self.initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))
        # save seeds and height_maps
        self.seeds_backup = copy.deepcopy(self.seeds)
        # save internal states
        self.facings_backup = copy.deepcopy(self.facings)
        self._step_counts_backup = self._step_counts.copy()
        self.all_rews_backup = self.all_rews.copy()
        self.all_dones_backup = self.all_dones.copy()
        # save attractor states
        self.attractor_states_backup = []
        for i in range(len(self.attractor_handles)):
            attractor_prop = self.gym.get_attractor_properties(self.envs[i], self.attractor_handles[i])
            self.attractor_states_backup.append(copy.deepcopy(attractor_prop.target))
        # backup reward_fifo
        self.reward_fifo_backup = copy.deepcopy(self.reward_fifo)

    # site recovery
    def resume(self):
        # recover physical states
        self.gym.set_sim_rigid_body_states(self.sim, self.initial_state, gymapi.STATE_ALL)
        # recover seeds and height_maps
        self.seeds = copy.deepcopy(self.seeds_backup)
        for i in range(self.num_matrix_envs):
            self.set_landscape(i, self.seeds[i], update_collection = False)
        # recover internal states
        self.facings = copy.deepcopy(self.facings_backup)
        self._step_counts = self._step_counts_backup.copy()
        self.all_rews = self.all_rews_backup.copy()
        self.all_dones = self.all_dones_backup.copy()
        # recover attractor states
        for i in range(len(self.attractor_handles)):
            self.gym.set_attractor_target(self.envs[i], self.attractor_handles[i], self.attractor_states_backup[i])
        self._compute_obs()
        # recover reward_fifo
        self.reward_fifo = copy.deepcopy(self.reward_fifo_backup)

    # reset wfc landscape
    def __resetActors(self, env_id):
        for color in self.color_name_list:
            for cube in self.all_cubes_handler[env_id][color]:
                if cube.used:
                    cube.used = False
                    self.__moveActor(env_id=env_id, actor=cube.handler, pos=self.initial_pos)
                    self.gym.set_actor_scale(self.envs[env_id], cube.handler, self.vanish_scale)
            for ramp in self.all_ramps_handler[env_id][color]:
                if ramp.used:
                    ramp.used = False
                    self.__moveActor(env_id=env_id, actor=ramp.handler, pos=self.initial_pos)
                    self.gym.set_actor_scale(self.envs[env_id], ramp.handler, self.vanish_scale)
            for corner in self.all_corners_handler[env_id][color]:
                if corner.used:
                    corner.used = False
                    self.__moveActor(env_id=env_id, actor=corner.handler, pos=self.initial_pos)
                    self.gym.set_actor_scale(self.envs[env_id], corner.handler, self.vanish_scale)
        if self.actor_handles[env_id]:
            self.__moveActor(env_id=env_id, actor=self.actor_handles[env_id], pos=self.initial_pos)
        if self.food_handles[env_id]:
            self.__moveActor(env_id=env_id, actor=self.food_handles[env_id], pos=self.initial_pos)

    # reset wfc landscape interface
    def reset_all_landscape(self):
        for env_id in range(self.num_matrix_envs):
            self.__resetActors(env_id)

    def __getPos(self, i, j):
        pose_ij = gymapi.Transform()
        # y-axis up, same as Unity3D and the original model's coordinal system 
        pose_ij.p = gymapi.Vec3(i * self.prefab_size,  j * self.prefab_size, 0)
        pose_ij.r = gymapi.Quat(0,0,0,1)
        return pose_ij

    def __moveActor(self, env_id, actor, pos):
        body_states = self.gym.get_actor_rigid_body_states(self.envs[env_id], self.food_handler, gymapi.STATE_POS)
        body_states['pose']['p'].fill((pos.p.x, pos.p.y, pos.p.z))
        body_states['pose']['r'].fill((pos.r.x, pos.r.y, pos.r.z, pos.r.w))
        self.gym.set_actor_rigid_body_states(self.envs[env_id], actor, body_states, gymapi.STATE_POS)

    def __applytoActor(self, env_id, tile_type, pos, cell_index, handler_list):
        # get a not used handle and use it
        for handle in handler_list:
            select_handle = None
            if not handle.used:
                select_handle = handle
                handle.used = True
                self.__moveActor(env_id, handle.handler, pos)
                # scale selected block to normal size
                self.gym.set_actor_scale(self.envs[env_id], handle.handler, 1)
                # add handler to objects pool
                self.grid_tile_blocks[env_id][cell_index].append(handle)
                break
        if select_handle is None:
            raise Exception(f"No more {tile_type} available")

    @staticmethod
    def __rotateAsMark(mark, name="cube"):
        if mark > 3:
            mark = 0
            print("Rotate data is out of range 0-3, fallback to 0")
        base_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        base_axis = gymapi.Vec3(0, 0, 1)
        if name == "corner":
            base_axis = gymapi.Vec3(1, 0, 0)
            base_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5*math.pi)
        elif name == "ramp":
            base_axis = gymapi.Vec3(0, 1, 0)
            base_angle = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -0.5*math.pi)
            mark = -mark
        # reverse for fastwfc support
        mark = -mark
        rotation = base_angle * gymapi.Quat.from_axis_angle(base_axis, mark * 0.5 * math.pi)
        return rotation
    
    def __gridRender(self, seed, env_id):
        if isinstance(seed, Wave):
            seed = seed.seed
        # create actors based on WFC seed
        cell_index = 0
        self.grid_tile_blocks[env_id] = {}
        for i in range(0,9):
            for j in range(0,9):
                # cell_index = i*9+j
                self.grid_tile_blocks[env_id][cell_index] = []
                tile_ = seed[i*9+j][0][0]
                rot = seed[i*9+j][0][1]
                pose_ij = self.__getPos(i, j)
                # 1 - 6 stacked cubes
                if 0<tile_<= 6:
                    for q in range(tile_):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                # 7 -21 corners
                # 7 - 11, corners with all cubes
                elif 7<=tile_<=11:
                    # Instantiate base cube at the position for high layer corners
                    # The bottom layer is a cube, so the corners tile will never be the bottom layer
                    for q in range(tile_ - 6):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                    # Instantiate the corners at the position
                    pose_ij.p.z = (tile_ - 6) * self.prefab_height
                    pose_ij.r = self.__rotateAsMark(rot, name="corner")
                    self.__applytoActor(env_id=env_id, tile_type="corner", pos=pose_ij, cell_index=cell_index, handler_list=self.all_corners_handler[env_id][self.color_name_list[tile_ - 6]])
                # 12 - 15, corners with 1 extra corner and cubes
                elif 12<=tile_<=15:
                    # Instantiate base cube at the position for high layer corners
                    # The bottom layer is a cube, so the corners tile will never be the bottom layer
                    for q in range(tile_ - 11):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                    for q in range(tile_ - 11-1, tile_-11+1):
                        # Instantiate the corners at the position
                        pose_ij.p.z = (q+1) * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot, name="corner")
                        self.__applytoActor(env_id=env_id, tile_type="corner", pos=pose_ij, cell_index=cell_index, handler_list=self.all_corners_handler[env_id][self.color_name_list[q+1]])
            #  16 - 18, corners with 2 extra corner and cubes
                elif 16<=tile_<=18:
                    # Instantiate base cube at the position for high layer corners
                    # The bottom layer is a cube, so the corners tile will never be the bottom layer
                    for q in range(tile_ - 15):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                    for q in range(tile_ - 15-1, tile_-15+2):
                        # Instantiate the corners at the position
                        pose_ij.p.z = (q+1) * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot, name="corner")
                        self.__applytoActor(env_id=env_id, tile_type="corner", pos=pose_ij, cell_index=cell_index, handler_list=self.all_corners_handler[env_id][self.color_name_list[q+1]])
            # 19 - 20, corners with 3 extra corner and cubes
                elif 19<= tile_ <= 20:
                    for q in range(tile_ - 18):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                    for q in range(tile_ - 18-1, tile_-18+3):
                        pose_ij.p.z = (q+1) * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot, name="corner")
                        self.__applytoActor(env_id=env_id, tile_type="corner", pos=pose_ij, cell_index=cell_index, handler_list=self.all_corners_handler[env_id][self.color_name_list[q+1]])
                #21, corners with 4 extra corner and cubes
                elif tile_ == 21:
                    for q in range(tile_ - 20):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                    for q in range(tile_ - 20-1, tile_-20+4):
                        pose_ij.p.z = (q+1) * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot, name="corner")
                        self.__applytoActor(env_id=env_id, tile_type="corner", pos=pose_ij, cell_index=cell_index, handler_list=self.all_corners_handler[env_id][self.color_name_list[q+1]])
                # 22 - 26 ramps
                elif 22<=tile_<=26:
                    # Instantiate bottom cubes first
                    for q in range(tile_ - 21):
                        pose_ij.p.z = q * self.prefab_height
                        pose_ij.r = self.__rotateAsMark(rot)
                        self.__applytoActor(env_id=env_id, tile_type="cube", pos=pose_ij, cell_index=cell_index, handler_list=self.all_cubes_handler[env_id][self.color_name_list[q]])
                    # Instantiate the ramps at the position
                    pose_ij.p.z = (tile_ - 21) * self.prefab_height
                    pose_ij.r = self.__rotateAsMark(rot, name="ramp")
                    self.__applytoActor(env_id=env_id, tile_type="ramp", pos=pose_ij, cell_index=cell_index, handler_list=self.all_ramps_handler[env_id][self.color_name_list[tile_ - 21]])
                else:
                    raise Exception("The tile Data is out of the range: 0 to 21")
                cell_index += 1

    # set wfc landscape interface
    def set_landscape(self, env_id, seed_, update_collection = True):
        if not isinstance(seed_, Wave):
            seed_ = Wave(seed_)
        # set seed
        self.seeds[env_id] = copy.deepcopy(seed_)

        # update collection of seeds
        if update_collection:
            landscape_new = True
            for s in self.seeds_collection:
                if s == seed_:
                    landscape_new = False
                    break
            if landscape_new:
                self.seeds_collection.append(copy.deepcopy(seed_))

            # print("collection : ", len(self.seeds_collection))
        seed = seed_.seed
        self.__resetActors(env_id)
        self.__gridRender(seed, env_id)
        self.placeAgentAndFood(env_id)

    def _apply_action(self, env_id, action):

        self._step_counts[env_id] += 1
        
        # print self.step_counts[env_id]
        # print('step_counts:', self._step_counts[env_id])

        # apply action to an environment

        i = env_id

        '''
        apply action
        '''
        body_states = self.gym.get_actor_rigid_body_states(self.envs[i], self.actor_handles[i], gymapi.STATE_ALL)
        # transform global coordinate system to local coordinate system
        body_ = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], 0)
        body_t = self.gym.get_rigid_transform(self.envs[i], body_)
        body_t.p = gymapi.Vec3(0,0,0)
        verts_ = body_t.transform_points(self.verts)

        # maintain vertical speed
        z_spd = body_states['vel']['linear'][0][2]

        if action == 0:
            body_states['vel']['linear'].fill((self.spd * verts_[1][0][0],self.spd * verts_[1][0][1],z_spd))
            self.gym.set_actor_rigid_body_states(self.envs[i], self.actor_handles[i], body_states, gymapi.STATE_VEL)
        elif action == 1:
            body_states['vel']['linear'].fill((-self.spd * verts_[1][0][0],-self.spd * verts_[1][0][1],z_spd))
            self.gym.set_actor_rigid_body_states(self.envs[i], self.actor_handles[i], body_states, gymapi.STATE_VEL)
        elif action == 2:
            body_states['vel']['linear'].fill((-self.spd * verts_[2][0][0],-self.spd * verts_[2][0][1],z_spd))
            self.gym.set_actor_rigid_body_states(self.envs[i], self.actor_handles[i], body_states, gymapi.STATE_VEL)
        elif action == 3:
            body_states['vel']['linear'].fill((self.spd * verts_[2][0][0],self.spd * verts_[2][0][1],z_spd))
            self.gym.set_actor_rigid_body_states(self.envs[i], self.actor_handles[i], body_states, gymapi.STATE_VEL)
        elif action == 4:
            self.facings[i] -= self.facing_step
            attractor_pose = gymapi.Transform()
            attractor_pose.p = gymapi.Vec3(0, 0.0, 0.0)
            attractor_pose.r = gymapi.Quat(0,1,0,1) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), self.facings[i]*math.pi)
            self.gym.set_attractor_target(self.envs[i], self.attractor_handles[i], attractor_pose)
        elif action == 5:
            self.facings[i] += self.facing_step
            attractor_pose = gymapi.Transform()
            attractor_pose.p = gymapi.Vec3(0, 0.0, 0.0)
            attractor_pose.r = gymapi.Quat(0,1,0,1) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), self.facings[i]*math.pi)
            self.gym.set_attractor_target(self.envs[i], self.attractor_handles[i], attractor_pose)

    def _apply_actions(self, all_actions):

        for i in range(self.num_envs):
            self._apply_action(env_id = i, action = all_actions[i])

    def _step_physics(self):

        '''
        step the physics
        '''
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless:
            self.gym.clear_lines(self.viewer)
        self.gym.step_graphics(self.sim)

    def _compute_obs(self):

        '''
        get observations
        '''
        self.gym.render_all_camera_sensors(self.sim)

        # self.vec_obs = np.zeros((self.num_matrix_envs, HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        for i in range(self.num_matrix_envs):

            # observation = np.zeros((HEIGHT,WIDTH, 4), np.uint8)
            rgb_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            # 1. transform the image from RGBA to RGB
            # 2. transpose the image from (HEIGHT,WIDTH, 3) to (3, HEIGHT, WIDTH)
            self.vec_obs[i] = np.copy(np.transpose(rgb_image.reshape((HEIGHT,WIDTH, 4))[:, :, :3], (2, 0, 1)))

    def _compute_rews(self):
        
        # self.all_rews = np.zeros(self.num_matrix_envs, dtype=np.float32)

        for i in range(self.num_matrix_envs):

            body_states_actor = self.gym.get_actor_rigid_body_states(self.envs[i], self.actor_handles[i], gymapi.STATE_ALL)
            actor_p = body_states_actor['pose']['p']
            body_states_food = self.gym.get_actor_rigid_body_states(self.envs[i], self.food_handles[i], gymapi.STATE_ALL)
            food_p = body_states_food['pose']['p']

            # calculate distance between actor and food
            ap = np.array([actor_p[0][0], actor_p[0][1], actor_p[0][2]])
            fp = np.array([food_p[0][0], food_p[0][1], food_p[0][2]])
            dist = np.linalg.norm(ap - fp)
            # print(dist)

            if dist < self.proximity_threshold:
                self.all_rews[i] = 1.0
            else:
                # sparse reward
                self.all_rews[i] = 0.0
                # dense reward 
                # self.all_rews[i] = (float)(self.proximity_threshold) / (float)(dist)
                # self.all_rews[i] = (float)(self.proximity_threshold) / (float)(dist+5)

        # # log the reward
        # for i in range(self.num_matrix_envs):
        #     self._rews[i] += self.all_rews[i]

        return np.copy(self.all_rews)

    def _compute_dones(self, log_rew = True):

        # self.all_dones = np.zeros(self.num_matrix_envs, dtype=np.bool8)
        for i in range(self.num_matrix_envs):

            self.all_dones[i] = False

            if self.all_rews[i] == 1.0:
                self.all_dones[i] = True
                continue
            
            body_states_actor = self.gym.get_actor_rigid_body_states(self.envs[i], self.actor_handles[i], gymapi.STATE_ALL)
            actor_p = body_states_actor['pose']['p']
            actor_cnnc_x = ((actor_p[0][0]+1)/(9*2))*20*9
            actor_cnnc_y = ((actor_p[0][1]+1)/(9*2))*20*9

            body_states_food = self.gym.get_actor_rigid_body_states(self.envs[i], self.food_handles[i], gymapi.STATE_ALL)
            food_p = body_states_food['pose']['p']
            food_cnnc_x = ((food_p[0][0]+1)/(9*2))*20*9
            food_cnnc_y = ((food_p[0][1]+1)/(9*2))*20*9

            if not (actor_cnnc_x >= 0 and actor_cnnc_x < 20*9 and actor_cnnc_y >= 0 and actor_cnnc_y < 20*9):
                self.all_dones[i] = True
                continue
            
            if not (food_cnnc_x >= 0 and food_cnnc_x < 20*9 and food_cnnc_y >= 0 and food_cnnc_y < 20*9):
                self.all_dones[i] = True
                continue

            if self._step_counts[i] > self.max_steps:
                self.all_dones[i] = True
                continue

        # log the rewards
        if log_rew :
            for i in range(self.num_matrix_envs):
                if self.all_dones[i] == True:
                    # compute general success rate
                    if self.all_rews[i] == 1.0:
                        self.reward_fifo.append(self.all_rews[i])
                    else:
                        self.reward_fifo.append(self.all_rews[i])
                    # compute success rate for each environment
                    if self.all_rews[i] == 1.0:
                        self.reward_archive[i].append(self.all_rews[i])
                    else:
                        self.reward_archive[i].append(self.all_rews[i])
                
                # compute mean reward for each environment
                reward_avg = 0
                for j in range(len(self.reward_archive[i])):
                    reward_avg += self.reward_archive[i][j]
                if len(self.reward_archive[i]) > 0:
                    reward_avg = reward_avg / len(self.reward_archive[i])
                
                self.rew_avg[i] = reward_avg
            

        return np.copy(self.all_dones)

    def _compute_infos(self):
        return [{} for _ in range(self.num_envs)]

    def _proximity(self, startx, starty, endx, endy):
            # calculate distance between actor and food
        ap = np.array([startx, starty])
        fp = np.array([endx, endy])
        dist = np.linalg.norm(ap - fp)

        if dist < self.proximity_threshold:
            return True
        else:
            return False

    def _path_achievable(self, env_id, startx, starty, endx, endy):

        return True 

    def reset(self, all_dones = None):
        # print("resetting...")
        if all_dones is not None:
            
            # reset envs that are done
            for i in range(self.num_envs):
                if all_dones[i]:

                    self._step_counts[i] = 0
                    self.facings[i] = 0
                    self.placeAgentAndFood(i)
                    self.all_dones[i] = False
        else:
            self.reward_fifo.clear()
            self.reward_fifo = deque(maxlen=300)
            self.best_performance = 0
            # reset all envs
            for i in range(self.num_envs):
                self._step_counts[i] = 0
                self.facings[i] = 0
                self.placeAgentAndFood(i)
                self.all_dones[i] = False

        if all_dones is None:
            self._compute_obs()
            return np.copy(self.vec_obs)

    def step(self, actions: np.ndarray, physics_only = False):

        # timer1 = time.time()
        # auto reset
        self.reset(all_dones = self.all_dones)

        # print("reset: ", time.time() - timer1)
        # timer1 = time.time()

        self._apply_actions(actions)

        # print("apply action: ", time.time() - timer1)
        # timer1 = time.time()

        self._step_physics()
        # print("step physics: ", time.time() - timer1)
        # timer1 = time.time()

        self._compute_obs()

        # print("compute obs: ", time.time() - timer1)

        all_rews = self._compute_rews()
        all_dones = self._compute_dones()
        all_infos = self._compute_infos()

        # print("compute obs: ", time.time() - timer1)
        # timer1 = time.time()

        # print(self.vec_obs)
        self.step_count += self.num_matrix_envs

        end_time = time.time()
        if (end_time - self.start_time) > 1:
            # print("FPS: %.2f" % ((self.step_count - self.last_frame_cnt)))
            self.last_frame_cnt = self.step_count
            self.start_time = time.time()

        return np.copy(self.vec_obs), np.copy(all_rews), np.copy(all_dones), all_infos

    def render(self, real_time = False):

        if not self.headless:

            if self.gym.query_viewer_has_closed(self.viewer):
                return -2

            for i in range(self.num_matrix_envs):
                body_states = self.gym.get_camera_transform(self.sim, self.envs[i], self.camera_handles[i])
                draw_camera(self.gym, self.viewer,self.envs[i],body_states)

            # update the viewer
            self.gym.draw_viewer(self.viewer, self.sim, True)

            if real_time:
                self.gym.sync_frame_time(self.sim)

    def evaluate_sb3_on_single_env(self, model, seed, num_episodes = 100, render = False):
        '''
        evaluate model performance on single map
        '''

        print("evaluating model performance on single map...")
        eval_start = time.time()
        
        # site reservation
        self.pause()

        env_sr_log = []

        # 1. apply landscape "seed" to all envs
        for i in range(self.num_matrix_envs):
            self.set_landscape(env_id = i, seed_ = seed, update_collection = False)
        # reset env pool
        observation_ = self.reset()

        # 2. evaluate model until num_episodes of done signals are collected
        while len(self.reward_fifo) < num_episodes:
            action_, _ = model.predict(observation_, deterministic=False)
            observation_, _, _, _ = self.step(action_)

        # 3. compute mean reward of reward_fifo
        reward_sum = 0
        for i in range(len(self.reward_fifo)):
            reward_sum += self.reward_fifo[i]
        reward_avg = reward_sum / len(self.reward_fifo)
        env_sr_log.append(reward_avg)

        # site recover
        self.resume()

        eval_end = time.time()
        print("evaluation time: %.2f" % (eval_end - eval_start))
        print("evaluation completed")

        # submit performance report
        return copy.deepcopy(env_sr_log)

    def evaluate_sb3(self, model, num_episodes = 100, render = False):

        '''
        evaluate model performance on map collection
        '''

        if render:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
            cam_pos = gymapi.Vec3(0.0, -15.0, 15.0)
            cam_target = gymapi.Vec3(4, 4, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        print("evaluating model performance on map collection...")
        eval_start = time.time()
        
        # site reservation
        self.pause()

        env_sr_log = []

        # traverse the last 13 elements of seeds_collection
        # evaluate model for each environment
        for si in range(min(len(self.seeds_collection), 13)):
            # 1. apply landscape "seed" to all envs
            i_start = min(len(self.seeds_collection), 13)
            for i in range(self.num_matrix_envs):
                self.set_landscape(env_id = i, seed_ = self.seeds_collection[len(self.seeds_collection)-(i_start-si)], update_collection = False)
            # reset env pool
            observation_ = self.reset()

            # 2. evaluate model until num_episodes of done signals are collected
            pbar = tqdm(total = num_episodes)
            last = 0
            print(f"{si}/ {min(len(self.seeds_collection), 13)}")
            while len(self.reward_fifo) < num_episodes:
                if len(self.reward_fifo) > last:
                    last = len(self.reward_fifo)
                    pbar.update(1)
                action_, _ = model.predict(observation_, deterministic=False)
                observation_, _, _, _ = self.step(action_)

                if render:
                    # update the viewer
                    self.gym.draw_viewer(self.viewer, self.sim, True)
            pbar.close()
            # 3. compute mean reward of reward_fifo
            reward_sum = 0
            for i in range(len(self.reward_fifo)):
                reward_sum += self.reward_fifo[i]
            reward_avg = reward_sum / len(self.reward_fifo)
            env_sr_log.append(reward_avg)

        # site recover
        self.resume()

        eval_end = time.time()
        print("evaluation time: %.2f" % (eval_end - eval_start))
        print("evaluation completed")

        if render:
            self.gym.destroy_viewer(self.viewer)

        # submit performance report
        return copy.deepcopy(env_sr_log)

    def close (self):
        
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False
