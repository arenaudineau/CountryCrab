###
# Copyright (2024) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import numpy as np
import pandas as pd
import os
import typing as t
import math

import campie
import cupy as cp


def walksat_m(architecture, config, params):

    tcam = architecture[0]
    ram = architecture[1]
    tcam_cores = architecture[2]
    ram_cores = architecture[3]

    # get parameters. This should be "fixed values"
    # max runs is the number of parallel initialization (different inputs)
    max_runs = params.get("max_runs", 100)
    # max_flips is the maximum number of iterations
    max_flips = params.get("max_flips", 1000)
    # noise profile
    noise_dist = params.get("noise_distribution",'normal')
    # number of cores
    n_cores = params.get("n_cores", 1)
    # variables
    variables = tcam.shape[1]


    # get configuration. This is part of the scheduler search space
    # noise is the standard deviation of noise applied to the make_values
    noise = config.get('noise',0.8)

    
    # note, to speed up the code violated_constr_mat does not represent the violated constraints but the unsatisfied variables. It doesn't matter for the overall computation of p_vs_t
    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    # generate random inputs
    inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)
    # tracks the amount of iteratiosn that are actually completed
    n_iters = 0

    for it in range(max_flips):
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

    return violated_constr_mat, n_iters, inputs
