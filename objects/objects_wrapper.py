import random
from datetime import datetime
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


class YcbObjects:

    def __init__(self, load_path, save_path, trials, special_cases=None):
        self.load_path = load_path
        with open(load_path + '/obj_list.txt') as f:
            lines = f.readlines()
            self.obj_names = [line.rstrip('\n') for line in lines]
        self.succes_rate = dict.fromkeys(self.obj_names, (0, 0))
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_dir = f'{save_path}/{now}'
        os.mkdir(self.save_dir)
        self.trials = trials
        self.special_cases = special_cases
            
    def shuffle_objects(self):
        random.shuffle(self.obj_names)
    
    def get_obj_path(self, obj_name):
        return f'{self.load_path}/Ycb{obj_name}/model.urdf'

    def add_succes_target(self, obj_name):
        current = self.succes_rate[obj_name]
        self.succes_rate[obj_name] = (current[0]+1, current[1])

    def add_succes_grasp(self, obj_name):
        current = self.succes_rate[obj_name]
        self.succes_rate[obj_name] = (current[0], current[1]+1)

    def print_succes(self):
        print("Successes:")
        for item, amount in self.succes_rate.items():
            print("{} target:({}) grasp:({})".format(item, amount[0], amount[1]))

    def write_json(self):
        data = json.dumps(self.succes_rate)
        f = open(self.save_dir+'/data.json', 'w')
        f.write(data)
        f.close()

    def plot(self):
        df = pd.DataFrame(self.succes_rate).T
        df.columns = ['target', 'grasped']
        df.plot(kind='bar')
        plt.xlabel('Name')
        plt.ylabel('N')
        plt.title(f'Objects succesfully grasped and placed in target | {self.trials} trials')
        plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(self.save_dir+'/plot.png')

    def write_summary(self):
        n_objects = len(self.obj_names)
        
        total_grasped = 0
        total_target = 0
        for item, amount in self.succes_rate.items():
            total_grasped += amount[1]
            total_target += amount[0]
        
        with open(self.save_dir+'/summary.txt', 'w') as f:
            f.write(f'Target acc={total_target / (self.trials * n_objects)}\n')
            f.write(f'Grasp acc={total_grasped / (self.trials * n_objects)}\n')

    def summarize_results(self):
        self.write_json()
        self.write_summary()
        self.plot()
