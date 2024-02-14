import numpy as np
import pandas as pd
import os
import re
import typing as t
import math

from compiler import map_camsat
from analyze import vector_tts

import campie
import cupy as cp


def camsat(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    # config contains parameters to optimize, params are fixed

    # Check GPUs are available.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        raise RuntimeError(
            f"No GPUs available. Please, set `CUDA_VISIBLE_DEVICES` environment variable."
        )
    #print('selected gpu')
    #print(os.environ.get("CUDA_VISIBLE_DEVICES", None))
    instance_addr = config["instance"]
    #print('loaded instance')
    tcam_array, ram_array = map_camsat(instance_addr)
    tcam_array = (1-tcam_array)
    #print('arrays compiled')
    max_runs = params.get("max_runs", 1000)

    clauses = tcam_array.shape[0]
    variables = tcam_array.shape[1]
    #print('starting campie initialization')
    inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)

    tcam = cp.asarray(tcam_array, dtype=cp.float32)
    ram = cp.asarray(ram_array, dtype=cp.float32)
    
    n_variables = tcam.shape[1]
    n_words = config.get("n_words", clauses)
    n_cores = config.get("n_cores", 1)
    scheduling = params.get("scheduling", "round_robin")

    task = params.get("task", "debug")

    if task == "solve":
        fname = params["hp_location"]
        optimized_hp = pd.read_csv(fname)
        if n_cores>1:
            filtered_df = optimized_hp[
                (optimized_hp["n_cores"] == n_cores)
                & (optimized_hp["n_words"] == n_words)
                & (optimized_hp["N_V"] == n_variables)
            ]
        else:
            filtered_df = optimized_hp[(optimized_hp["N_V"] == n_variables)]            
        noise = filtered_df["noise"].values[0]
        max_flips = int(filtered_df["max_flips_max"].values[0])
        max_flips_median = int(filtered_df["max_flips_median"].values[0])

    else:
        noise = config.get("noise", 0.5)
        max_flips = params.get("max_flips", 1000)

    if scheduling == "fill_first":
        needed_cores = math.ceil(tcam.shape[0] / n_words)
        if n_cores < needed_cores:
            raise ValueError(
                f"Not enough CAMSAT cores available for mapping the instance: clauses={clauses}, n_cores={n_cores}, n_words={n_words}, needed_cores={needed_cores}"
            )

        # potentially reduce the amount of cores used to the actually needed amount
        n_cores = needed_cores

        # extend tcam and ram so they can be divided by n_cores
        if clauses % n_cores != 0:
            padding = n_cores * n_words - tcam.shape[0]
            tcam = cp.concatenate(
                (tcam, cp.full((padding, variables), cp.nan)), dtype=cp.float32
            )
            ram = cp.concatenate(
                (ram, cp.full((padding, variables), 0)), dtype=cp.float32
            )

    elif scheduling == "round_robin":
        core_size = math.ceil(tcam.shape[0] / n_cores)

        # create potentialy uneven splits, that's why we need a python list
        tcam_list = cp.array_split(tcam, n_cores)
        ram_list = cp.array_split(ram, n_cores)

        # even out the sizes of each core via padding
        for i in range(len(tcam_list)):
            if tcam_list[i].shape[0] == core_size:
                continue

            padding = core_size - tcam_list[i].shape[0]
            tcam_list[i] = cp.concatenate(
                (tcam_list[i], cp.full((padding, variables), cp.nan)), dtype=cp.float32
            )
            ram_list[i] = cp.concatenate(
                (ram_list[i], cp.full((padding, variables), 0)), dtype=cp.float32
            )

        # finally, update the tcam and ram, with the interspersed padding now added
        tcam = cp.concatenate(tcam_list)
        ram = cp.concatenate(ram_list)

    else:
        raise ValueError(f"Unknown scheduling algorithm: {scheduling}")

    # split into cores
    tcam_cores = tcam.reshape((n_cores, -1, variables))
    ram_cores = ram.reshape((n_cores, -1, variables))

    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    # tracks the amount of iteratiosn that are actually completed
    n_iters = 0

    for it in range(max_flips - 1):
        n_iters += 1

        # global
        matches = campie.tcam_match(inputs, tcam)
        hd = campie.tcam_hamming_distance(inputs, tcam)
        y = matches @ ram

        violated_constr = cp.sum(y > 0 , axis=1)
        violated_constr_mat[:, it] = violated_constr

        # early stopping
        if cp.sum(violated_constr_mat[:, it]) == 0:
            break

        if n_cores == 1:
            # there is no difference between the global matches and the core matches
            # if there is only one core. we can just copy the global results and
            # and wrap a single core dimension around them
            matches, y, violated_constr = map(
                lambda x: x[cp.newaxis, :],
                [matches, y, violated_constr],
            )
        else:
            # otherwise, actually compute the matches for each core
            matches = campie.tcam_match(inputs, tcam_cores)
            y = matches @ ram_cores
            violated_constr = cp.sum(y > 0, axis=2)

        # add noise
        y += noise * cp.random.randn(*y.shape, dtype=y.dtype)

        # select highest values
        update = cp.argmax(-y, axis=2)
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
    

    p_vs_t = cp.sum(violated_constr_mat[:, 1 : n_iters + 1] == 0, axis=0) / max_runs
    p_vs_t = cp.asnumpy(p_vs_t)
    violated_constr_mat = cp.asnumpy(violated_constr_mat)

    if task == "hpo":
        if np.sum(p_vs_t) > 0:
            tts = vector_tts(
                np.linspace(1, len(p_vs_t) + 1, len(p_vs_t)), p_vs_t, p_target=0.99
            )
            best_tts = np.min(tts[tts > 0])
            best_max_flips = np.where(tts == tts[tts > 0][np.argmin(tts[tts > 0])])
            return {"tts": best_tts, "max_flips_opt": best_max_flips[0][0]}

        else:
            return {"tts": np.nan, "max_flips_opt": max_flips}

    elif task == "debug":
        inputs = cp.asnumpy(inputs)
        return p_vs_t, violated_constr_mat,inputs

    else:
        if "pipeline" in params and params["pipeline"] == True:
            p_vs_t = np.clip(p_vs_t * 3, 0, 1)

        if np.sum(p_vs_t) > 0:
            tts = vector_tts(
                np.linspace(1, len(p_vs_t) + 1, len(p_vs_t)), p_vs_t, p_target=0.99
            )
            # check if p_max = 1
            if np.max(p_vs_t) == 1:
                # check if 1 before the max_flips
                idx = np.where(p_vs_t==1)[0][0]
                if idx<max_flips_median:
                    tts_median = np.where(p_vs_t>=0.99)[0][0]
                else:
                    tts_median = tts[max_flips_median]
            else:
                tts_median = tts[max_flips_median]
                    
            tts_max = tts[-2]

            return {"tts_max": tts_max, "tts_median": tts_median, "p_max":np.max(p_vs_t)}

        else:
            return {"tts_max": np.nan, "tts_median": np.nan}
