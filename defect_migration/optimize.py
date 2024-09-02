import argparse
import os
import shutil
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pymatgen.core import Structure, Lattice, PeriodicSite
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.local_env import CrystalNN

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hyperopt as hy
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn import ALIGNN
from jarvis.analysis.structure.spacegroup import Spacegroup3D

class DefectOptimization:
    def __init__(self):
        self.algo_step = 0
        self.structure_num = 0
        os.makedirs('results/cif_files', exist_ok=True)
        with open(os.path.join('results', 'energy_data.csv'), 'w+') as f:
            f.writelines("step, energy,time\n")
        
        self.start_time = time.time()
        self.structure = Structure.from_file("POSCAR.cif")
        self.cd_sites = [site for site in self.structure if site.species_string == "Cd"]
        self.zn_sites = [site for site in self.structure if site.species_string == "Zn"]
        self.te_sites = [site for site in self.structure if site.species_string == "Te"]
        self.as_sites = [site for site in self.structure if site.species_string == "As"]
        self.vacancy_site = random.choice(self.as_sites)
        self.vacancy_index = self.as_sites.index(self.vacancy_site)
        #self.structure.remove_sites([self.vacancy_index])
        #self.writer = CifWriter(self.structure)
        #self.writer.write_file("Cd_vacancy.cif")
        self.nn_algorithm = CrystalNN(weighted_cn=True, distance_cutoffs=None)
        self.nearest_neighbours = self.nn_algorithm.get_nn_info(self.structure, self.vacancy_index)
        
        self.optimization_algo(200, 500, 100, algorithm='anneal')
    
    def create_structure(self, kwargs):
        distorted_structure = self.structure.copy()
        dict_vals = {}
        self.structure_num += 1
        
        for i in range(len(self.nearest_neighbours)): 
            dict_vals[i] = [
                float(kwargs[f'x{i}']),
                float(kwargs[f'y{i}']),
                float(kwargs[f'z{i}'])
            ]
        
        step = 0
        for neighbor in self.nearest_neighbours:
            new_fractional_coords = Lattice(distorted_structure.lattice.matrix).get_fractional_coords(dict_vals[step])
            new_site = PeriodicSite(neighbor['site'].species, new_fractional_coords, distorted_structure.lattice)
            distorted_structure = Structure.from_sites([new_site if idx == neighbor['site_index'] else site for idx, site in enumerate(distorted_structure)])
            step += 1
        
        new_cif_file = f"results/cif_files/predict{self.structure_num}.cif"
        writer = CifWriter(distorted_structure)
        writer.write_file(new_cif_file)
        with open(os.path.join('results/cif_files', 'id_prop.csv'), 'w+') as f:
            f.writelines(f"predict{self.structure_num}, {0}")
    
    def predict_formation_energy(self, kwargs):
        self.algo_step += 1
        self.create_structure(kwargs)
        cifpath = 'results/cif_files'
        results = f'results/results{self.algo_step}'
        device = "cpu"
        model = ALIGNN()
        model.load_state_dict(torch.load("checkpoint_90.pt")["model"])
        model.to(device)
        model.eval()

        def predict_energy(cif_file):
            atoms = Atoms.from_cif(cif_file)
            cvn = Spacegroup3D(atoms).conventional_standard_structure

            g, lg = Graph.atom_dgl_multigraph(atoms)
            out_data = (
                model([g.to(device), lg.to(device)])
                .detach()
                .cpu()
                .numpy()
                .flatten()
                .tolist()[0]
            )
            return out_data
        
        energy = predict_energy(f'results/cif_files/predict{self.structure_num}.cif')
        dict_final = {'loss': energy, 'status': 'ok'}

        with open(os.path.join('results', 'energy_data.csv'), 'a+') as f:
            f.write(','.join([str(self.algo_step),
                              str(energy),
                              str(time.time() - self.start_time)]) + '\n')
        
        return dict_final
    
    def optimization_algo(self, n_init, max_step, rand_seed, algorithm="bayesian"):
        pbounds = {}
        
        for i in range(len(self.nearest_neighbours)):
            pbounds[f'x{i}'] = hy.hp.uniform(f'x{i}', self.nearest_neighbours[i]['site'].coords[0] - 0.2, self.nearest_neighbours[i]['site'].coords[0] + 0.2)
            pbounds[f'y{i}'] = hy.hp.uniform(f'y{i}', self.nearest_neighbours[i]['site'].coords[1] - 0.2, self.nearest_neighbours[i]['site'].coords[1] + 0.2)
            pbounds[f'z{i}'] = hy.hp.uniform(f'z{i}', self.nearest_neighbours[i]['site'].coords[2] - 0.2, self.nearest_neighbours[i]['site'].coords[2] + 0.2)

        if algorithm == 'rand':
            print('using Random Search ...')
            algo = hy.rand.suggest
        elif algorithm == 'anneal':
            print('using Simulated Annealing ...')
            algo = hy.partial(hy.anneal.suggest)
        elif algorithm == 'bayesian':
            print('using Bayesian Optimization ...')
            algo = hy.partial(hy.tpe.suggest, n_startup_jobs=n_init)
        else:
            print("Invalid Algorithm")

        if rand_seed == -1:
            rand_seed = None
        else:
            rand_seed = np.random.default_rng(rand_seed)

        trials = hy.Trials()
        best = hy.fmin(fn=self.predict_formation_energy,
                       space=pbounds,
                       algo=algo,
                       max_evals=max_step,
                       trials=trials,
                       rstate=rand_seed)
        print(best)

if __name__ == "__main__":
    optimizer = DefectOptimization()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = '/depot/amannodi/data/Habibur_Projects/Defects_in_Cd_Zn_S_Se_Te_Project_1/ALIGNN_Training/alignn_optimization/results/energy_data.csv'
data = pd.read_csv(file_path)

# Convert energy from eV to meV
data.iloc[:, 1] = data.iloc[:, 1] * 1000

# Sort data by energy from highest to lowest
#data = data.sort_values(by=data.columns[1], ascending=False)

# Reset the index to ensure proper plotting
data = data.reset_index(drop=True)

# Scatter plot with enhanced visualization
plt.figure(figsize=(7, 6))

plt.scatter(x=data.index, y=data.iloc[:, 1], color='blue', s=40, marker='o', edgecolor='blue')

plt.xlabel('Optimization Step', fontsize=16)
plt.ylabel('Crystal Formation Energy/atom (meV)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Simulated Annealing', fontsize=16)

# Adjust the layout to remove excess blank space
plt.tight_layout()

# Save the plot
plt.savefig('Anneal_energy_vs_step.png', dpi=300)

plt.show()






