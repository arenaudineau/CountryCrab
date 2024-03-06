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

def count_variables(list_of_lists):
    # Flatten the list of lists
    flattened_list = [abs(item) for sublist in list_of_lists for item in sublist]
    # Get largest integer in the list
    largest_integer = max(flattened_list)
    # Return the largest integer. This in the feature can be changed to the actual number of variables
    return largest_integer

def compile_walksat_m(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    
    instance_name = config["instance"]
    clauses_list = load_clauses_from_cnf(instance_name)
    
    clauses = len(clauses_list)
    variables = count_variables(clauses_list)
    # map clauses to TCAM
    tcam_array = np.zeros([clauses, variables])    
    tcam_array[:] = np.nan
    for i in range(len(clauses_list)):
        tcam_array[i,abs(np.array(clauses_list[i]))-1]=clauses_list[i]
    tcam_array[tcam_array>0] = 1
    tcam_array[tcam_array<0] = 0
    # map clauses to RAM
    ram_array = tcam_array*1
    ram_array[ram_array==0]=1
    ram_array[np.isnan(ram_array)]=0


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
    #formula = CNF(from_file=instance_name)
    # extract clauses
    #clauses = list(filter(None, formula.clauses))

    instance_name = config["instance"]
    clauses_list = load_clauses_from_cnf(instance_name)
    #print(clauses)
    #for clause in clauses:
    #    solver.add_clause(clause)
    #clauses = np.array(list(filter(None, formula.clauses)))
    #clauses = np.array(clauses)

    clauses = len(clauses_list)
    variables = count_variables(clauses_list)

    # map clauses to TCAM
    ramf_array = np.zeros([2*variables,clauses])
    ramb_array = np.zeros([clauses,2*variables])
    
    for i,clause in enumerate(clauses_list):
        for literal in clause:
            if literal>0:
                ramf_array[2*(literal-1),i]=1
                ramb_array[i,2*(literal-1)]=1
            else:
                ramf_array[2*(-literal-1)+1,i]=1
                ramb_array[i,2*(-literal-1)+1]=1    
    architecture = [ramf_array, ramb_array]
    return architecture, params


# QUBO mappings available

# Returns W, B, C of the QUBO energy from 3-SAT cnf (k = 3 SAT only so far)
# W is returned as a symmetric matrix, i.e. E = x^T*W*x/2 + Bx + C
    
# mapping type 1: "clause_wise" mapping introduces an auxiliary variable for each clause of the cnf (Total QUBO size N = N_var + N_clauses)
# mapping type 2: "shared" mapping introduces each auxiliary varibale for multiple clauses by solving an optimization 
#     problem of finding x_ix_j = y substitutions minimizing the total number of aux variables (Total QUBO size N < N_var + N_clauses)
#
# mapping name 1: "Rosenberg" penalty (observed to be better for parallel updates on uf 3-SAT)
# mapping name 2: "KZFD" penalty      (observed to be better for single updates on uf 3-SAT)

def qubo_3sat_map(config: t.Dict) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    instance_name = config["instance"]
    clauses = load_clauses_from_cnf(instance_name)
    for c in clauses:
        if len(c) != 3:
            raise RuntimeError("3 sat only!")

    mapping_type = config["mapping_type"]   #clause_wise or shared   
    if not (mapping_type in ["clause_wise", "shared"]):
        raise RuntimeError("wrong mapping type!")
    
    mapping_name = config["mapping_name"]   #Rosenberg or KZFD    
    if not (mapping_name in ["Rosenberg", "KZFD"]):
        raise RuntimeError("wrong mapping name!") 
    
    if mapping_type == "clause_wise":
        W, B, C = clause_wise_qubo_3sat_map(clauses, mapping_name)
    else:
        W, B, C = shared_qubo_3sat_map(clauses, mapping_name)

    return W, B, C


def shared_qubo_3sat_map(clauses: t.List[t.List[int]], mapping_name) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    num_vars = len(np.unique(abs(np.array(clauses))))
    num_clauses = len(clauses)

    cl = -np.array(clauses)            #invert the clauses to map to energy
    cl_idx =  np.abs(cl) - np.full_like(cl, 1) #count the variables from 0

    W1 = np.zeros((num_vars, num_vars))
    B1 = np.zeros(num_vars)
    C = np.zeros(1)
    
    for i in range(num_clauses):
        if cl[i, 0] > 0: 
            if cl[i, 1] > 0: 
                if cl[i, 2] > 0:
                    pass
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] += 1

            else:
                if cl[i, 2] > 0:
                    W1[cl_idx[i,0], cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] -= 1
                    W1[cl_idx[i,0], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,0]] += 1
            
        else:
            if cl[i,  1] > 0: 
                if cl[i, 2] > 0: 
                    W1[cl_idx[i,1], cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] -= 1
                    W1[cl_idx[i,1], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,1]] += 1
                
            else:
                if cl[i,2] > 0: 
                    W1[cl_idx[i,0], cl_idx[i,2]] -= 1
                    W1[cl_idx[i,1], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] += 1
                    W1[cl_idx[i,0], cl_idx[i,2]] += 1
                    W1[cl_idx[i,1], cl_idx[i,2]] += 1

                    B1[cl_idx[i,0]] -= 1
                    B1[cl_idx[i,1]] -= 1
                    B1[cl_idx[i,2]] -= 1

                    C += 1 
    
    count3 = np.full(num_clauses, 1)

    #take into account repeating 3rd order terms
    for i in range(num_clauses):
        if count3[i] == 0:
            continue
        c1 = np.sort(cl_idx[i, :])

        first_match = True
        for j in range(i+1, num_clauses):
            if count3[j] == 0:
                continue
            c2 = np.sort(cl_idx[j, :])
            if (c1 == c2).all():
                if first_match:
                    if np.prod(cl[i, :]) > 0:
                        count3[i] = 1
                    else:
                        count3[i] = -1
                    first_match = False
                
                if np.prod(cl[j, :]) > 0:
                    count3[i] += 1
                else:
                    count3[i] -= 1
                
                count3[j] = 0

    pairs_matrix = np.zeros((num_vars, num_vars), dtype=int)
    for i in range(num_clauses):
        if count3[i] != 0:
            pairs_matrix[cl_idx[i, 0], cl_idx[i, 1]] += 1
            pairs_matrix[cl_idx[i, 0], cl_idx[i, 2]] += 1
            pairs_matrix[cl_idx[i, 1], cl_idx[i, 2]] += 1
    
    pairs_matrix += np.transpose(pairs_matrix)

    pair_count = []
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            if pairs_matrix[i, j] > 0:
                pair_count.append([np.array([i, j]), pairs_matrix[i, j]])

    pair_count = sorted(pair_count, key = lambda p: p[1], reverse=True)

    shared_pairs = []
    track_clauses = np.full(num_clauses, 1)

    for i in range(num_clauses):
        if i == 45:
            pass
        if track_clauses[i] == 0 or count3[i] == 0:
            continue
        
        c01 = np.sort([cl_idx[i, 0], cl_idx[i, 1]])
        c02 = np.sort([cl_idx[i, 0], cl_idx[i, 2]])
        c12 = np.sort([cl_idx[i, 1], cl_idx[i, 2]])

        for k in range(len(pair_count)):
            save_pair = False

            if (c01 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c02 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1

            if (c02 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c01 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1
                        
            if (c12 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c01 == pair_count[l][0]).all() or (c02 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1

            if save_pair:
                pair_count[k][1] -= 1
                
                for j in range(i+1, num_clauses):
                    if count3[j] == 0 or track_clauses[j] == 0:
                        continue

                    c01 = np.sort([cl_idx[j, 0], cl_idx[j, 1]])
                    c02 = np.sort([cl_idx[j, 0], cl_idx[j, 2]])
                    c12 = np.sort([cl_idx[j, 1], cl_idx[j, 2]])

                    if (c01 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c02 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                    if (c02 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c01 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                    if (c12 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c01 == pair_count[l][0]).all() or (c02 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                shared_pairs.append(pair_count[k][0])
                if pair_count[k][1] != 0:
                    raise RuntimeError("Error")
                
                pair_count = sorted(pair_count, key = lambda p: p[1], reverse=True)

                break
    
    N = num_vars + len(shared_pairs)

    W = np.zeros((N, N))
    B = np.zeros(N)

    W[:num_vars, :num_vars] = W1
    B[:num_vars] = B1

    if mapping_name == "Rosenberg":
        penalty_lb = np.zeros((len(shared_pairs), 3))

    check = 0
    for i in range(num_clauses):
        c01 = np.sort([cl_idx[i, 0], cl_idx[i, 1]])
        c02 = np.sort([cl_idx[i, 0], cl_idx[i, 2]])
        c12 = np.sort([cl_idx[i, 1], cl_idx[i, 2]])
    
        found_pair = False
        for k in range(len(shared_pairs)):
            if (c01 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 2], num_vars + k] += 1
                else:
                    W[cl_idx[i, 2], num_vars + k] -= 1

            if (c02 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 1], num_vars + k] += 1
                else:
                    W[cl_idx[i, 1], num_vars + k] -= 1

            if (c12 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 0], num_vars + k] += 1
                else:
                    W[cl_idx[i, 0], num_vars + k] -= 1
    
            if found_pair:
                check += 1
                if mapping_name == "Rosenberg":
                    if np.prod(cl[i, :]) > 0:
                        penalty_lb[k, 0] += 1
                    else:
                        penalty_lb[k, 1] += 1
                    
                    if penalty_lb[k, 2] <  np.max(penalty_lb[k, :]):
                        W[shared_pairs[k][0], shared_pairs[k][1]] += 1
                        W[shared_pairs[k][0], num_vars + k] -= 2
                        W[shared_pairs[k][1], num_vars + k] -= 2
                        B[num_vars + k] += 3

                        penalty_lb[k, 2] += 1

                elif mapping_name == "KZFD":
                    W[shared_pairs[k][0], shared_pairs[k][1]] += 1
                    W[shared_pairs[k][0], num_vars + k] -= 1
                    W[shared_pairs[k][1], num_vars + k] -= 1
                    B[num_vars + k] += 1
                    
                    if np.prod(cl[i, :]) < 0:
                        B[num_vars + k] += 1
                        W[shared_pairs[k][0], shared_pairs[k][1]] -= 1

                else:
                    raise RuntimeError("Unknown mapping!")
                
                break
    # print(f"substituted {check} clauses")
    W += np.transpose(W)

    return W, B, C


def clause_wise_qubo_3sat_map(clauses: t.List[t.List[int]], mapping_name) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_vars = len(np.unique(abs(np.array(clauses))))
    num_clauses = len(clauses)

    N = num_vars + num_clauses

    cl = -np.array(clauses)            #invert the clauses to map to energy
    cl_idx =  np.abs(cl) - np.full_like(cl, 1) #count the variables from 0

    W = np.zeros((N, N))
    B = np.zeros(N)
    C = np.zeros(1)

    for i in range(num_clauses):
      if cl[i, 2] > 0:
        W[num_vars + i, cl_idx[i, 2]] = 1
      else:
        B[num_vars + i] = 1
        W[num_vars + i, cl_idx[i, 2]] = -1

    for i in range(num_clauses):
      if mapping_name == "Rosenberg":
          B[num_vars + i] += 3
      else:
          B[num_vars + i] += 1

      if cl[i, 0] > 0 and cl[i, 1] > 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] += 1

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 0]] -= 2
              W[num_vars + i, cl_idx[i, 1]] -= 2
          else:
              W[num_vars + i, cl_idx[i, 0]] -= 1
              W[num_vars + i, cl_idx[i, 1]] -= 1

      elif cl[i, 0] > 0 and cl[i, 1] < 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] -= 1
          B[cl_idx[i, 0]] += 1

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 0]] -= 2
              W[num_vars + i, cl_idx[i, 1]] += 2
              B[num_vars + i] -= 2
          else:
              W[num_vars + i, cl_idx[i, 0]] -= 1
              W[num_vars + i, cl_idx[i, 1]] += 1
              B[num_vars + i] -= 1

      elif cl[i, 0] < 0 and cl[i, 1] > 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] -= 1
          B[cl_idx[i, 1]] += 1

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 1]] -= 2
              W[num_vars + i, cl_idx[i, 0]] += 2
              B[num_vars + i] -= 2
          else:
              W[num_vars + i, cl_idx[i, 1]] -= 1
              W[num_vars + i, cl_idx[i, 0]] += 1
              B[num_vars + i] -= 1

      else:
          W[cl_idx[i, 0], cl_idx[i, 1]] += 1
          B[cl_idx[i, 0]] -= 1
          B[cl_idx[i, 1]] -= 1
          C += 1  # add the constant

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 0]] += 2
              W[num_vars + i, cl_idx[i, 1]] += 2
              B[num_vars + i] -= 4
          else:
              W[num_vars + i, cl_idx[i, 0]] += 1
              W[num_vars + i, cl_idx[i, 1]] += 1
              B[num_vars + i] -= 2

    W += W.transpose()   

    return W, B, C
