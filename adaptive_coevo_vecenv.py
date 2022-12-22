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
from sympy import false
import cv2
from utils import tileid_to_json
from draw import *
import json
import copy
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


if __name__ == "__main__":

    LOGDIR = "./training_logs"
    # timesteps = 1500000
    timesteps = 200

    m_env = PCGVecEnv(headless_ = False)
    m_env.reset()

    model_ref = None
    check_freq = 1000
    best_step = 0

    wfcworker_ = fastwfc.XLandWFC("samples.xml")

    # evaluation callback
    def evaluate_cb(local_vars, global_vars):
        
        if not m_env.headless:
            # render during training
            m_env.render()


    model = PPO('CnnPolicy', env=m_env, batch_size = 1024)
    # model.set_random_seed(3407)
    model_ref = model

    print("Training boost model")

    # 1. boost training
    # model.learn(total_timesteps=timesteps, callback=evaluate_cb)

    print("Training boost model done")
    print("Co-Evolution starts ...")

    # 6. outter loop
    for g in range(50):

        qc_pass = False

        while not qc_pass:

            # 2. inner loop of coevolution
            num_decendents = 10

            # generational results
            model_duplicates = []
            map_decendents = []
            performance_records = []

            # duplicate the model
            for i in range(num_decendents):
                model_ = PPO('CnnPolicy', env=m_env, batch_size = 1024)
                model_.set_parameters(model.get_parameters())
                # model_.set_random_seed(3407)
                model_duplicates.append(model_)

            for i in range(num_decendents):
                print("Starting decendent evaluation -------------------------------- ", i)

                # 2.1. generate new map decendent
                new_decendent = m_env.generate_decendent()

                # update decendents database
                map_decendents.append(copy.deepcopy(new_decendent))

                print("new decendent generated : ", new_decendent.seed)
                
                # 2.2. seed distribution
                m_env.seed_distribution()

                # print("landscape layout : ", m_env.seeds)

                # 2.3. larva evaluation
                larva_eval = m_env.evaluate_sb3(model = model_duplicates[i], num_episodes = 300, render=True)

                print("larva evaluation : ", larva_eval)

                # 2.4. evaluate map decendent by training
                # model_duplicates[i].learn(total_timesteps=timesteps, callback=evaluate_cb)

                # 2.5. adult evaluation
                adult_eval = m_env.evaluate_sb3(model = model_duplicates[i], num_episodes = 300, render=True)

                print("adult evaluation : ", adult_eval)

                # update performance log
                performance_records.append([copy.deepcopy(larva_eval), copy.deepcopy(adult_eval)])

                # 2.6. reversion of map collection
                print("reversion of map collection : ", len(m_env.seeds_collection))
                m_env.revert_map_collection()
                print("reversion of map collection done : ", len(m_env.seeds_collection))

            print("generational training completed")

            # 3. evaluate fitness of each decendent
            evaluation = []
            for i in range(len(performance_records)):

                larva_eval = performance_records[i][0]
                adult_eval = performance_records[i][1]

                # 3.1. maximum performance drop on old maps
                sr_drop_max = -1000
                for j in range(len(larva_eval)-1):
                    sr_drop = larva_eval[j] - adult_eval[j]
                    if sr_drop > sr_drop_max:
                        sr_drop_max = sr_drop

                # 3.2. performance gain on new maps
                sr_gain = adult_eval[-1] - larva_eval[-1]

                # 3.3. success rate on new maps
                sr_new = adult_eval[-1]

                evaluation.append(copy.deepcopy([sr_drop_max, sr_gain, sr_new]))

            # 4. quality control
            old_sr_threshold = 0.8
            new_sr_threshold = 0.86
            qc_list = []
            for e in range(len(performance_records)):
                _adult_eval = performance_records[e][1]
                # 4.1 minimum success rate on old maps
                min_sr = 100000
                for i in range(len(_adult_eval)-1):
                    if _adult_eval[i] < min_sr:
                        min_sr = _adult_eval[i]
                # 4.2 success rate on new maps
                sr_new = _adult_eval[-1]
                if min_sr > old_sr_threshold and sr_new > new_sr_threshold:
                    qc_list.append(True)
                else:
                    qc_list.append(False)

            qc_pass = False
            for qc in qc_list:
                if qc:
                    qc_pass = True
                    break

            if qc_pass:
                print("Quality Control passed : ", min_sr, sr_new)
            else:
                print("Quality Control failed : ", min_sr, sr_new)
                print("run inner loop again......")

            # select best candidate as winner
            winner_id = 0
            winner_score = 10000000
            for i in range(len(evaluation)):
                score = sqrt((evaluation[i][0]+1)**2 + (evaluation[i][1]-1)**2 + (evaluation[i][2]-1)**2)
                print("seed : ", i, " score : ", score)
                if qc_list[i] and score < winner_score:
                    winner_score = score
                    winner_id = i

            print("--------------------winner id-------------------- ", winner_id)
            print("-------------------winner score------------------ ", winner_score)

            # 5. update Collection and Model
            if qc_pass:
                # 5.1. update map collection
                m_env.seeds_collection.append(copy.deepcopy(map_decendents[winner_id]))
                # 5.2. update model
                model_ref.set_parameters(model_duplicates[winner_id].get_parameters())
                # 5.3. save map_decendents
                for m in range(len(map_decendents)):
                    file_name = "gen_"+ str(g) + "_dec_" + str(m) + ".json"
                    tileid_to_json(map_decendents[m].wave, save_path=file_name)
                # 5.4. save model
                model_duplicates[winner_id].save("./training_logs/model_gen_" + str(g))
                # 5.5. save performance_records
                with open("performance_records_gen_" + str(g) + ".json", "w") as f:
                    json.dump(performance_records, f)
