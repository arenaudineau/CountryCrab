import numpy as np
from pysat.solvers import Minisat22
from pysat.formula import CNF

def get_instance_names(path_to_dataset, k = 3, V = [20,50,75,100,150]):
    if k == 3:
        all_datasets_name = ['uf20-91', 'uf50-218', 'uf75-325', 'uf100-430', 'uf125-538', 'uf150-645', 'uf175-753', 'uf200-860', 'uf225-960', 'uf250-1065']
        all_variables_size = np.array([20, 50, 75, 100, 125, 150, 175, 200, 225, 250])
        set_1 = np.linspace(901,1000,100).astype(int)
        set_1_data = [20,50,100]
        set_2 = np.linspace(1,100,100).astype(int)
        #V = [20,50,75,100,150]
        I_V_all = {20: []}
        I_V_final = {20: []}
        I_V_opt = {20: []}
        for n_var in range(len(V)):
            N_V = V[n_var]
            dataset_name = all_datasets_name[np.where(all_variables_size == N_V)[0][0]]    
            I_V_all[N_V] = []
            if N_V in set_1_data:
                for instance in set_1:
                    instance_name = '/uf'+str(N_V)+'-0'+str(instance)+'.cnf'
                    instance_addr = path_to_dataset+dataset_name+instance_name
                    I_V_all[N_V].append(instance_addr)
            else:
                for instance in set_2:
                    instance_name = '/uf'+str(N_V)+'-0'+str(instance)+'.cnf'
                    instance_addr = path_to_dataset+dataset_name+instance_name
                    I_V_all[N_V].append(instance_addr)  
            I_V_all_card = len(I_V_all[N_V])
            I_V_opt_card = int(0.2*I_V_all_card)
            I_V_opt[N_V] = I_V_all[N_V][0:I_V_opt_card]
            I_V_final[N_V] = I_V_all[N_V][I_V_opt_card:I_V_all_card]
    elif k == 4:
        all_datasets_name = ['uf50-499','uf100-988','uf150-1492']
        V = np.array([50,100,150])
        I_V_all = {V[0]: []}
        I_V_final = {V[0]: []}
        I_V_opt = {V[0]: []}

        print('in the case of 4-SAT we are currently benchmarking the first 100 instances')
        set = np.linspace(1,100,100).astype(int)

        for n_var in range(len(V)):
            N_V = V[n_var]
            dataset_name = all_datasets_name[np.where(V == N_V)[0][0]]    
            I_V_all[N_V] = []
            for instance in set:
                if instance<10:
                    instance_name = '/uf'+str(N_V)+'-00'+str(instance)+'.cnf'
                    instance_addr = path_to_dataset+dataset_name+instance_name
                    I_V_all[N_V].append(instance_addr)
                else:
                    instance_name = '/uf'+str(N_V)+'-0'+str(instance)+'.cnf'
                    instance_addr = path_to_dataset+dataset_name+instance_name
                    I_V_all[N_V].append(instance_addr)
                I_V_all_card = len(I_V_all[N_V])
                # 20% of the instances used for HPO
                I_V_opt_card = int(0.2*I_V_all_card)
                I_V_opt[N_V] = I_V_all[N_V][0:I_V_opt_card]
                I_V_final[N_V] = I_V_all[N_V][I_V_opt_card:I_V_all_card]

    return I_V_opt, I_V_final


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