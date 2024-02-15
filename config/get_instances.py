import numpy as np

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
