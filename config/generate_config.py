import json
import os
import sys
import numpy as np
# Add the CountryCrab folder to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_instances import get_instance_names

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
I_V_opt, I_V_final = get_instance_names(path, k=3)

data = {
    # Architecture parameters
    "n_words": 128,
    "n_cores": 20,
    "scheduling": "fill_first",
    # Heuristic parameters
    "max_runs": 1000,
    "max_flips": 1000,
    "noise": 0.5,
    "noise_distribution": "normal",
    # Experiments parameter
    "p_solve": 0.99,
    "task": "debug",
    "hp_location": "/home/pedretti/projects/camsat/camsat_v2/data/experiments/f2f_hpo_3SAT.csv",
    "experiment_name": "exp2_3sat_uniform",
    "instance_list": I_V_final,
}

# Specify the filename
filename = "debug_hierarchical.json"

# Writing JSON data
with open(filename, 'w') as f:
    json.dump(data, f, indent=4)
