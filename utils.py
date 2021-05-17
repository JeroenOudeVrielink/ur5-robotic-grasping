import random
from datetime import datetime
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


class YcbObjects:

    def __init__(self, load_path, save_path, trials ,special_cases=None):
        self.load_path = load_path
        self.trials = trials
        with open(load_path + '/obj_list.txt') as f:
            lines = f.readlines()
            self.obj_names = [line.rstrip('\n') for line in lines]
        self.succes_target = dict.fromkeys(self.obj_names, 0)
        self.succes_grasp = dict.fromkeys(self.obj_names, 0)
        self.tries = dict.fromkeys(self.obj_names, 0)
        self.special_cases = special_cases
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_dir = f'{save_path}/{now}'
        os.mkdir(self.save_dir)

    def shuffle_objects(self):
        random.shuffle(self.obj_names)
    
    def get_obj_path(self, obj_name):
        return f'{self.load_path}/Ycb{obj_name}/model.urdf'

    def add_succes_target(self, obj_name):
        self.succes_target[obj_name] += 1

    def add_succes_grasp(self, obj_name):
        self.succes_grasp[obj_name] += 1

    def add_try(self, obj_name):
        self.tries[obj_name] += 1

    def print_succes(self):
        print("Successes per object:")
        for obj in self.obj_names:
            tries = self.tries[obj]
            target = self.succes_target[obj]
            grasp = self.succes_grasp[obj]
            print(f'{obj}: Target acc={target/tries} ({target}/{tries}) Grasp acc={grasp/tries} ({grasp}/{tries})')
        total_tries = sum(self.tries.values())
        total_target = sum(self.succes_target.values())
        total_grasp = sum(self.succes_grasp.values())
        print('Total:')
        print(f'Target acc={total_target/total_tries} ({total_target}/{total_tries}) Grasp acc={total_grasp/total_tries} ({total_grasp}/{total_tries})')

    def write_json(self):
        data_tries = json.dumps(self.tries)
        data_target = json.dumps(self.succes_target)
        data_grasp = json.dumps(self.succes_grasp)
        f = open(self.save_dir+'/data_tries.json', 'w')
        f.write(data_tries)
        f.close()        
        f = open(self.save_dir+'/data_target.json', 'w')
        f.write(data_target)
        f.close()        
        f = open(self.save_dir+'/data_grasp.json', 'w')
        f.write(data_grasp)
        f.close()

def plot(path, tries, target, grasp, trials):
    succes_rate = dict.fromkeys(tries.keys())
    for obj in succes_rate.keys():
        acc_target = target[obj] / tries[obj]
        acc_grasp = grasp[obj] / tries[obj]
        succes_rate[obj] = (acc_target, acc_grasp)
    df = pd.DataFrame(succes_rate).T
    df.columns = ['Target', 'Grasped']
    df = df.sort_values(by ='Target', ascending=True)
    ax = df.plot(kind='bar')
    plt.xlabel('Name')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy objects placed in target and grasped | {trials} trials')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.3)

    plt.savefig(path+'/plot.png')

def write_summary(path, tries, target, grasp):        
    with open(path+'/summary.txt', 'w') as f:
        total_tries = sum(tries.values())
        total_target = sum(target.values())
        total_grasp = sum(grasp.values())
        f.write('Total:\n')
        f.write(f'Target acc={total_target/total_tries:.3f} ({total_target}/{total_tries}) Grasp acc={total_grasp/total_tries:.3f} ({total_grasp}/{total_tries})\n')
        f.write('\n')
        f.write("Accuracy per object:\n")
        for obj in tries.keys():
            n_tries = tries[obj]
            n_t = target[obj]
            n_g = grasp[obj]
            f.write(f'{obj}: Target acc={n_t/n_tries:.3f} ({n_t}/{n_tries}) Grasp acc={n_g/n_tries:.3f} ({n_g}/{n_tries})\n')

def summarize(path, trials):
    with open(path+'/data_tries.json') as data:
        tries = json.load(data)    
    with open(path+'/data_target.json') as data:
        target = json.load(data)    
    with open(path+'/data_grasp.json') as data:
        grasp = json.load(data)
    plot(path, tries, target, grasp, trials)
    write_summary(path, tries, target, grasp)


if __name__=='__main__':
    path = 'results/isolated_obj_percentages_15-5'
    summarize(path, trials=20)