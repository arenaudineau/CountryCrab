from compiler import map_camsat
import numpy as np
import pandas as pd

def compute_energy(instance_path):
    energy_list = []
    for instance in instance_path:
        tcam_array, ram_array = map_camsat(instance)
        energy_consumption = tcam_energy(ram_array) + dpe_energy(ram_array)
        energy_list.append({'config/instance': instance, 'CAMSAT energy [J]': energy_consumption})
        
    energy_df = pd.DataFrame(energy_list)
    return energy_df

def tcam_energy(tcam):
    num_clauses = tcam.shape[0]
    num_variables = tcam.shape[1]

    # Energy model for the TCAM array, based on the number of clauses and variables for
    # a specific SAT problem in cnf formulation.

    # 16 nm modeled with CACTI, data from Cat Graves' T-NANO paper https://ieeexplore.ieee.org/abstract/document/8812926
    c_sl_cell = 0.12e-15
    c_driver = 2.14e-15
    c_sl = c_sl_cell * num_clauses + c_driver
    c_ml_cell = 0.19e-15
    c_precharge = 0.23e-15
    c_mlso = 1.85e-15
    c_ml = c_ml_cell * num_variables + c_precharge + c_mlso
    
    # compute the order of each clause. The maximum number of discharges is equal to the number of variables in a clause
    order = np.sum(tcam,axis = 1)
    c_ml_3 = c_ml_cell * order + c_precharge + c_mlso
    
    v_sl = 0.8
    v_ml = 0.6
    energy_tcam = (
        num_variables * c_sl * v_sl**2
        + 0.5 * num_clauses * c_ml * v_ml**2
        + 0.5 * np.sum(c_ml_3) * v_ml**2
    )
    return energy_tcam   

def dpe_energy(ram_array):
    """
    Energy model for a dot-product engine (DPE) based on a specified DPE matrix.

    The matrix has shape variables x clauses.
    """
    #ram_array_pos = ram_array * (ram_array > 0)
    #ram_array_pos = ram_array_pos / np.max(ram_array_pos)
    #ram_array_neg = -ram_array * (ram_array < 0)
    #ram_array_neg = ram_array_neg / np.max(ram_array_neg)
    g_low = 1e-7
    # probably 2e-6 is the good to get the required noise
    g_high = 2e-6
    v_read = 0.1
    f_clk = 1e9

    ram_array[ram_array==0] = g_low
    ram_array[ram_array==1] = g_high

    energy_ram_array = (
        np.sum(
            v_read**2 * np.ones(ram_array.shape[1]) * ram_array
        )
        / f_clk
    )
    return energy_ram_array


