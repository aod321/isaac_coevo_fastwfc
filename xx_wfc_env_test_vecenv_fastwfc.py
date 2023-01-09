from asyncio import FastChildWatcher
from code import interact
import os
import time
import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
#from isaacgym import gymtorch
from math import sqrt
import math
import cv2
from draw import *

import fastwfc

from vec_env_fastwfc import PCGVecEnv, Wave

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import torch

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--wfc_size', type=int, default=9)
argparser.add_argument('--enable_node_pairs', type=bool, default=False)
args = argparser.parse_args()

WFC_SIZE = args.wfc_size
LOGDIR = "./training_logs"
ENBALE_NODE_PAIRS = args.enable_node_pairs

# m_env = PCGVecEnv(headless_ = False, compute_device_id=0, graphics_device_id=0, wfc_size=WFC_SIZE)
m_env = PCGVecEnv(wfc_size=WFC_SIZE, is_node_pairs=ENBALE_NODE_PAIRS, return_all=True, num_envs=1, headless_ = False, compute_device_id=0, graphics_device_id=0)
m_env.reset()

print("num_envs:",m_env.num_envs)

all_obs = np.zeros((16, 84, 84, 4), dtype=np.uint8)

all_actions = np.zeros(m_env.num_matrix_envs)
for i in range(m_env.num_matrix_envs):
    all_actions[i] = -2

steps_ = 0
reset_id = 0

wfcworker_ = fastwfc.XLandWFC(f"samples_{WFC_SIZE}{WFC_SIZE}.xml")

manual = False

m_env.pause()

while True:

    # steps_ += 1

    observation, reward, done, info = m_env.step(all_actions)
    if np.nonzero(reward)[0].size > 0:
        print("reward:",reward)
    obs1 = observation[0]
    # transpose obs1 from (3,84,84) to (84,84,3)
    obs1 = np.transpose(obs1, (1,2,0))

    # create a black opencv image of size(84,84)
    img = np.zeros((84,84,3), np.uint8)
    # display img
    cv2.imshow('img', obs1)
    key_ = cv2.waitKey(1)
    if key_ == 119:  # w
        action = 0
    elif key_ == 115:    # s
        action = 1
    elif key_ == 97:     # a
        action = 2
    elif key_ == 100:    # d
        action = 3
    elif key_ == 107:    # k ccw
        action = 4
    elif key_ == 108:    # l cw
        action = 5
    elif key_ == 105:    # i reset
        m_env.reset()
    elif key_ == 114:    # r resume
        m_env.resume()
        manual = True
    elif key_ == 112:    # p pause
        m_env.pause()
    elif key_ == 109:    # m change landscape
        seed, img = wfcworker_.generate(out_img=True)
        m_env.set_landscape(env_id = 0,seed_ = seed)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('map', img_bgr)
        cv2.waitKey(1)
    elif key_ == 118:    # v
        m_env.seed_distribution()
    else:
        action = -2

    # print(key_)

    for i in range(m_env.num_matrix_envs):
        all_actions[i] = action

    # # if manual:
    # #     key_ = cv2.waitKey(0)
    # #     if key_ == 109:
    # #         manual = False

    m_env.render()
