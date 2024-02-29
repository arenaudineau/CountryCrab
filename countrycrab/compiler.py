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
from pysat.solvers import Minisat22
from pysat.formula import CNF
import typing as t
import math
import cupy as cp

def load_clauses_from_cnf(file_path: str) -> t.List[t.List[int]]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        clauses = []
        for line in lines:
            if line.startswith('c') or line.startswith('p'):
                continue
            elif line.startswith('%'):
                break
            clause = [int(x) for x in line.strip().split() if x != '0']
            clauses.append(clause)
    return clauses

def compile_walksat_m(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    
    instance_name = config["instance"]
    clauses = load_clauses_from_cnf(instance_name)
    # map clauses to TCAM
    tcam_array = np.zeros([len(clauses), len(np.unique(abs(np.array(clauses))))])    
    tcam_array[:] = np.nan
    for i in range(len(clauses)):
        tcam_array[i,abs(np.array(clauses[i]))-1]=clauses[i]
    tcam_array[tcam_array>0] = 1
    tcam_array[tcam_array<0] = 0
    # map clauses to RAM
    ram_array = tcam_array*1
    ram_array[ram_array==0]=1
    ram_array[np.isnan(ram_array)]=0

    clauses = tcam_array.shape[0]
    variables = tcam_array.shape[1]

    tcam = cp.asarray(tcam_array, dtype=cp.float32)
    ram = cp.asarray(ram_array, dtype=cp.float32)
    # number of clauses that can map to each core
    n_words = params.get("n_words", clauses)
    n_cores = params.get("n_cores", 1)
    scheduling = params.get("scheduling", "fill_first")

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

    # rewrite the parameters
    params['n_cores'] = n_cores
    params['variables'] = variables
    params['clauses'] = clauses
    
    architecture = [tcam, ram, tcam_cores, ram_cores, n_cores]
    return architecture, params


def compile_walksat_g(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    instance_name = config["instance"]
    # print("Compiling instance with walksat_g...")
    # simple mapping, takes the instance and map it to a 'large' tcam and ram
    # load instance
    formula = CNF(from_file=instance_name)
    # extract clauses
    clauses = list(filter(None, formula.clauses))
    #for clause in clauses:
    #    solver.add_clause(clause)
    clauses = np.array(list(filter(None, formula.clauses)))
    # map clauses to TCAM
    ramf_array = np.zeros([2*len(np.unique(np.abs(clauses))),clauses.shape[0]])
    ramb_array = np.zeros([clauses.shape[0],2*len(np.unique(np.abs(clauses)))])
    for i in range(clauses.shape[0]):
        pos_literal_indices = np.where(clauses[i,]>0)[0]
        neg_literal_indices = np.where(clauses[i,]<0)[0]
        if len(pos_literal_indices)!=0:
            ramf_array[2*(np.abs(clauses[i,pos_literal_indices])-1),i]=1
            ramb_array[i,2*(np.abs(clauses[i,pos_literal_indices])-1)]=1
        if len(neg_literal_indices)!=0:
            ramf_array[2*(np.abs(clauses[i,neg_literal_indices])-1)+1,i]=1
            ramb_array[i,2*(np.abs(clauses[i,neg_literal_indices])-1)+1]=1
    
    architecture = [ramf_array, ramb_array]
    return architecture, params