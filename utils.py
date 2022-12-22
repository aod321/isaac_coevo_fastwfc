# date: 2022-11-24
# author: Yin Zi
# Some helper functions

import json
import copy

def tileid_to_json(tile_id: list, save_path: str):
    json_data ={}
    wave_list = []
    for i in range(81):
        wave_list.append(tile_id[i][0])
        wave_list.append(tile_id[i][1])
    json_data["data"] = wave_list
    json.dump(json_data, open(save_path, 'w'))
    return json_data

def json_to_tileid(filename):
    with open(filename) as json_file:
        json_file = json.load(json_file)
        data =copy.deepcopy(json_file['data'])
    data1 = []
    for i in range(len(data)//2):
        data1.append([data[2*i], data[2*i+1]])
    return data1
