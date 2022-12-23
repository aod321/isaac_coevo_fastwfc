import isaacgym
import torch
from asyncio import FastChildWatcher
from code import interact
from collections import deque
import os
import time
from tkinter.tix import Tree
import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
#from isaacgym import gymtorch
from math import sqrt
import math
from sympy import false
import cv2
from draw import *
import fastwfc
import multiprocessing
import copy
from utils import tileid_to_json
from vec_env_fastwfc import PCGVecEnv, Wave
import json
from datetime import datetime
from WFCUnity3DEnv_fastwfc import WFCUnity3DEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import ray
import torch

import asyncio

from multiprocessing import Process, Queue, Pipe, Manager, Value, Array, Lock, Event, Pool
import threading


def in_collection(seed,seeds_collection):
    for s in seeds_collection:
        if s == seed:
            return True
    return False

def train_model_on_collection(
    decendent_id, 
    model_parameters, 
    seeds_collection, 
    timesteps, 
    compute_device_id = 0, 
    graphics_device_id = 0, 
    cuda_id = 0, 
    return_queue = None
    ):


    # 1. create environment
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    m_env = PCGVecEnv(headless_ = True, compute_device_id = compute_device_id, graphics_device_id = graphics_device_id)
    m_env.reset()

    # 2. create model
    model = PPO('CnnPolicy', env=m_env, batch_size = 1024, device = 'cuda:'+str(cuda_id))
    model.set_parameters(copy.deepcopy(model_parameters))

    # 3. set collection of m_env
    m_env.update_collection(seeds_collection)

    # 4. seed distribution
    m_env.seed_distribution()
    m_env.reset()

    # 5. larva evaluation
    print("start larva evaluation on device : ", graphics_device_id)
    print(f"evaluating {decendent_id}")
    larva_eval = m_env.evaluate_sb3(model = model, num_episodes = 300)
    print("larva evaluation : ", larva_eval, " on device : ", graphics_device_id)

    # 6. evaluate map decendent by training
    print("start training on device : ", graphics_device_id)
    model.learn(total_timesteps=timesteps)
    print("training finished on device : ", graphics_device_id)

    print("start adult evaluation on device : ", graphics_device_id)
    print(f"evaluating {decendent_id}")
    # 7. adult evaluation
    adult_eval = m_env.evaluate_sb3(model = model, num_episodes = 300)
    print("adult evaluation : ", adult_eval, " on device : ", graphics_device_id)

    # 8. create model_cpu and transfer parameters
    model_cpu = PPO('CnnPolicy', env=m_env, batch_size = 1024, device = 'cpu:0')
    model_cpu.set_parameters(model.get_parameters())

    print("model transfered to cpu")

    # release VecEnv
    m_env.close()

    return_queue.put((
    decendent_id,
    copy.deepcopy(model_cpu.get_parameters()),
    copy.deepcopy(larva_eval),
    copy.deepcopy(adult_eval)
    ))

    return 0

def save_model_proc(model_parameters, generation_id):

    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    LOGDIR = "./training_logs"

    m_env = PCGVecEnv(headless_ = True)
    m_env.reset()
    # save model to LOGDIR
    model = PPO('CnnPolicy', env=m_env, batch_size = 1024, device = 'cuda:0')
    model.set_parameters(copy.deepcopy(model_parameters))
    model.save(LOGDIR + "/model_" + str(generation_id))

    print("model saved to : ", LOGDIR + "/model_" + str(generation_id))

def save_model(model_parameters, generation_id):

    p = Process(target=save_model_proc, args=(model_parameters, generation_id))
    p.start()
    p.join()
    p.kill()


def generate_boost_model(return_queue = None):

    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'

    LOGDIR = "./training_logs"
    timesteps = 1500000

    m_env = PCGVecEnv(headless_ = True)
    m_env.reset()

    # # check if LOGDIR+"/boost_model" exists
    # boost_model_exists = os.path.exists(LOGDIR+"/boost_model.zip")
    # if boost_model_exists:
    #     print("boost model exists, loading...")
    #     # load boost model
    #     model = PPO.load(LOGDIR+"/boost_model.zip", device = 'cpu:0')
    # else:
    model = PPO('CnnPolicy', env=m_env, batch_size = 1024, device = 'cuda:0')
    # evaluate larva
    print("generate_boost: Evaluating larva model")
    larva_eval = m_env.evaluate_sb3(model = model, num_episodes = 300)
    print("Larva eval: ", larva_eval)

    print("Training boost model")
    # boost training
    model.learn(total_timesteps=timesteps)

    # evaluate boost model
    print("Evaluating boost model")
    adult_eval = m_env.evaluate_sb3(model = model, num_episodes = 300)
    print("Adult eval: ", adult_eval)

    model_cpu = PPO('CnnPolicy', env=m_env, batch_size = 1024, device = 'cpu:0')
    model_cpu.set_parameters(model.get_parameters())
    return_queue.put(copy.copy(model_cpu.get_parameters()))

    print(model_cpu.get_parameters())

    # save model to LOGDIR
    model.save(LOGDIR + "/boost_model")

    m_env.close()

    print("boost model training complete and saved")

if __name__ == "__main__":


    LOGDIR = "./training_logs"
    timesteps = 1500000

    # load flat landscape
    wfcworker_ = fastwfc.XLandWFC("samples.xml")
    wave = wfcworker_.get_ids_from_wave(wfcworker_.build_a_open_area_wave())
    seed = Wave(wave)
    seeds_collection = deque(maxlen=64)         # update the seeds collection (winners) in outter loop
    model_collection = deque(maxlen=64)         # update the model collection in outter loop
    seeds_collection.append(seed)

    check_freq = 1000
    best_param = None   # parameters of the best model

    # 1. boost training
    boost_param = None

    boost_model_manager = multiprocessing.Manager()
    boost_model_queue = boost_model_manager.Queue()

    boost_model_process = Process(target=generate_boost_model, args=(boost_model_queue,))
    boost_model_process.start()
    boost_model_process.join()

    if boost_model_queue.empty() == False:
        print("boost model generate complete")
        boost_param = copy.copy(boost_model_queue.get())
        boost_model_queue.task_done()
    
    boost_model_manager.shutdown()
    boost_model_process.kill()

    best_param = boost_param

    print("boost param : ", boost_param)

    print("Training boost model done")
    print("Co-Evolution starts ...")

    # timesteps = 2000000

    # 6. outter loop
    for g in range(50):

        qc_pass = False

        while not qc_pass:

            manager = multiprocessing.Manager()
            return_q = manager.Queue()

            results = []

            # 2. inner loop of coevolution
            num_decendents = 8

            # generational results
            model_params = []
            map_decendents = []
            performance_records = []
            seeds_collection_duplicates = []

            # generate new seeds
            for i in range(num_decendents):
                base_wave = seeds_collection[-1].wave
                new_decendent, _ = wfcworker_.mutate(base_wave=wfcworker_.wave_from_id(base_wave), new_weight=162, iter_count=1, out_img=False)
                while in_collection(seed = Wave(new_decendent), seeds_collection = seeds_collection):
                    new_decendent, _ = wfcworker_.mutate(base_wave=wfcworker_.wave_from_id(base_wave), new_weight=162, iter_count=1, out_img=False)
                    new_decendent = Wave(new_decendent)
                new_decendent = Wave(new_decendent)
                # new seed is generated
                seeds_collection_duplicate = copy.deepcopy(seeds_collection)
                seeds_collection_duplicate.append(new_decendent)
                seeds_collection_duplicates.append(seeds_collection_duplicate)
                # update decendents database
                map_decendents.append(copy.deepcopy(new_decendent))
                print("new decendent generated : ", new_decendent.seed)

            # 3. train model-collection pairs on remote processes

            # 3.1 batch1 : 1-8 of 8 decendents
            p0 = Process(target=train_model_on_collection, args=(0, best_param, seeds_collection_duplicates[0], timesteps, 0,0,0,return_q,))
            p1 = Process(target=train_model_on_collection, args=(1, best_param, seeds_collection_duplicates[1], timesteps, 1,1,1,return_q,))
            p2 = Process(target=train_model_on_collection, args=(2, best_param, seeds_collection_duplicates[2], timesteps, 2,2,2,return_q,))
            p3 = Process(target=train_model_on_collection, args=(3, best_param, seeds_collection_duplicates[3], timesteps, 3,3,3,return_q,))
            p4 = Process(target=train_model_on_collection, args=(4, best_param, seeds_collection_duplicates[4], timesteps, 4,4,4,return_q,))
            p5 = Process(target=train_model_on_collection, args=(5, best_param, seeds_collection_duplicates[5], timesteps, 5,5,5,return_q,))
            p6 = Process(target=train_model_on_collection, args=(6, best_param, seeds_collection_duplicates[6], timesteps, 6,6,6,return_q,))
            p7 = Process(target=train_model_on_collection, args=(7, best_param, seeds_collection_duplicates[7], timesteps, 7,7,7,return_q,))
            
            p0.start()
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()

            p0.join()
            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            p6.join()
            p7.join()

            p0.kill()
            p1.kill()
            p2.kill()
            p3.kill()
            p4.kill()
            p5.kill()
            p6.kill()
            p7.kill()

            # 3.1.1 fetch results from return_q
            output_1 = []
            for i in range(return_q.qsize()):
                results = copy.copy(return_q.get())
                return_q.task_done()
                output_1.append(results)

            # 3.1.2 re-orgainze output_1 by output_1[:][0]
            output_1 = sorted(output_1, key=lambda x: x[0])

            # 3.4 merge output_1, output_2, output_3
            output_ = output_1

            print("--------------------- generation ", g, " training completed ---------------------")

            # print output_
            for i in range(len(output_)):
                print("G_",i,"_larva :",output_[i][2])
                print("G_",i,"_adult :",output_[i][3])

                # update trained model parameters
                model_params.append(copy.copy(output_[i][1]))
                # update performance log
                performance_records.append([copy.deepcopy(output_[i][2]), copy.deepcopy(output_[i][3])])

            # 4. evaluate fitness of each decendent
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

            # 5. quality control
            old_sr_threshold = 0.53
            new_sr_threshold = 0.65
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
                if min_sr >= old_sr_threshold and sr_new >= new_sr_threshold:
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
                # score = sqrt((evaluation[i][0]+1)**2 + (evaluation[i][1]-1)**2 + (evaluation[i][2]-1)**2)
                # score = abs(evaluation[i][0]+1) + abs(evaluation[i][1]-1) + abs(evaluation[i][2]-1)
                score = abs(evaluation[i][1]-1)
                print("seed : ", i, " score : ", score)
                if qc_list[i] and score < winner_score:
                    winner_score = score
                    winner_id = i

            
            # 6. update Collection and Model
            if qc_pass:

                print("--------------------winner id-------------------- ", winner_id)
                print("-------------------winner score------------------ ", winner_score)

                # 6.1. update map collection
                seeds_collection.append(copy.deepcopy(map_decendents[winner_id]))
                # 6.2. update model
                best_param = copy.copy(model_params[winner_id])
                # 6.3. save map_decendents
                json_save_path = "generated_maps/jsons/"
                time_tmp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                json_save_path = os.path.join(json_save_path, time_tmp)
                os.makedirs(json_save_path, exist_ok=True)
                # tag winner id via a empty folder
                os.makedirs(f"generated_maps/gen_{g}_winner_{winner_id}", exist_ok=True)
                for m in range(len(map_decendents)):
                    file_name = "gen_"+ str(g) + "_dec_" + str(m) + ".json"
                    file_name = os.path.join(json_save_path, file_name)
                    tileid_to_json(map_decendents[m].wave, save_path=file_name)
                # 6.4. save best model
                print("saving best model")
                save_model(best_param, g)
                
                # 6.5. save performance_records
                with open(os.path.join(json_save_path, "performance_records_gen_" + str(g) + ".json"), "w") as f:
                    json.dump(performance_records, f)
            else:
                print("Quality Control failed, incresing training timesteps")
                timesteps = timesteps + 300000
                timesteps = min(timesteps, 2000000)
                print("new timesteps : ", timesteps)


