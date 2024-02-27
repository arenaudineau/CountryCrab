import numpy as np
import pandas as pd
import os
import typing as t
import math

import campie
import cupy as cp


def walksat_m(data):
    inputs = data.get('inputs')   
    tcam = data.get('tcam')
    ram = data.get('ram')
    violated_constr_mat = data.get('violated_constr_mat')
    n_cores = data.get('n_cores')

    max_flips = data.get('max_flips')
    noise_dist = data.get('noise_dist')
    noise = data.get('noise')
    
# tracks the amount of iteratiosn that are actually completed
    n_iters = 0

    for it in range(max_flips - 1):
        n_iters += 1

        # global
        violated_clauses = campie.tcam_match(inputs, tcam)
        make_values = violated_clauses @ ram
        
        violated_constr = cp.sum(make_values > 0, axis=1)
        violated_constr_mat[:, it] = violated_constr

        # early stopping
        if cp.sum(violated_constr_mat[:, it]) == 0:
            break

        if n_cores == 1:
            # there is no difference between the global matches and the core matches (i.e. violated_clauses)
            # if there is only one core. we can just copy the global results and
            # and wrap a single core dimension around them
            violated_clauses, make_values, violated_constr = map(
                lambda x: x[cp.newaxis, :],
                [violated_clauses, make_values, violated_constr],
            )
        else:
            # otherwise, actually compute the matches (violated_clauses) for each core
            violated_clauses = campie.tcam_match(inputs, tcam_cores)
            make_values = violated_clauses @ ram_cores
            violated_constr = cp.sum(make_values > 0, axis=2)
        
        if noise_dist == 'normal':
            # add gaussian noise to the make values
            make_values += noise * cp.random.randn(*make_values.shape, dtype=make_values.dtype)  
        elif noise_dist == 'uniform':
            # add uniform noise. Note that the standard deviation is modulated by sqrt(3)
            make_values += cp.random.uniform(low=-noise*np.sqrt(3), high=noise*np.sqrt(3), size=make_values.shape, dtype=make_values.dtype) 
        elif noise_dist == 'intrinsic':
            # add noise considering automated annealing. Noise comes from memristor devices
            make_values += noise * cp.sqrt(make_values) * cp.random.randn(*make_values.shape, dtype=make_values.dtype)
        else:
            raise ValueError(f"Unknown noise distribution: {noise_dist}")

        # select highest values
        update = cp.argmax(make_values, axis=2)
        update[cp.where(violated_constr == 0)] = -1

        if n_cores == 1:
            # only one core, no need to do random picks
            update = update[0]
        else:
            # reduction -> randomly selecting one update
            update = update.T
            random_indices = cp.random.randint(0, update.shape[1], size=update.shape[0])
            update = update[cp.arange(update.shape[0]), random_indices]

        # update inputs
        campie.flip_indices(inputs, update[:, cp.newaxis])

    return violated_constr_mat, n_iters
