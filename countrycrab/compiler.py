import numpy as np
from pysat.solvers import Minisat22
from pysat.formula import CNF


def map_camsat(instance_name):
    # simple mapping, takes the instance and map it to a 'large' tcam and ram
    # load instance
    formula = CNF(from_file=instance_name)
    # extract clauses
    solver = Minisat22()
    clauses = list(filter(None, formula.clauses))
    for clause in clauses:
        solver.add_clause(clause)
    clauses = list(filter(None, formula.clauses))
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

    return tcam_array, ram_array

def map_camsat_g(instance_name):
    # simple mapping, takes the instance and map it to a 'large' tcam and ram
    # load instance
    formula = CNF(from_file=instance_name)
    # extract clauses
    solver = Minisat22()
    clauses = list(filter(None, formula.clauses))
    for clause in clauses:
        solver.add_clause(clause)
    clauses = list(filter(None, formula.clauses))
    # map clauses to TCAM
    tcam_array = np.zeros([len(clauses), len(np.unique(abs(np.array(clauses))))])    
    tcam_array[:] = np.nan
    for i in range(len(clauses)):
        tcam_array[i,abs(np.array(clauses[i]))-1]=clauses[i]
    tcam_array[tcam_array>0] = 1
    tcam_array[tcam_array<0] = 0
    # map clauses to RAM
    ram_array_pos = tcam_array*1
    ram_array_pos[np.isnan(ram_array_pos)]=0
    ram_array_neg = tcam_array*1
    ram_array_neg[ram_array_neg==1]=0
    ram_array_neg[ram_array_neg==0]=1
    ram_array_neg[np.isnan(ram_array_neg)]=0
    ram_array = np.empty((ram_array_pos.shape[0], ram_array_pos.shape[1] + ram_array_neg.shape[1]))
    ram_array[:, 0::2] = ram_array_pos
    ram_array[:, 1::2] = ram_array_neg
    
    return tcam_array, ram_array