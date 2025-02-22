# optimize k-means network in 3h resolution

# original network optimization
import pypsa
import os
import gurobipy as gp
import gc 
import pandas as pd
import numpy as np
import average_time_data

gc.collect()
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/optimized_kmeans_network_folder"
path = "/mnt/qb/work/ludwig/lqb072/Thesis4/elec_s.nc"

n = pypsa.Network(path) # to test averaging later

# cluster the network by k-means
bus_weightings = pd.Series(1, index=n.buses.index)
n_average = n.cluster.cluster_spatially_by_kmeans(n_clusters = 16, bus_weightings=bus_weightings)  # kmeans network to be averaged and optimized

# average the time data 
average_time_data.average_snapshots(n_average, time_window=3) 
# test if the average is correct 
print("Loads are correctly averaged: ", sum(n.loads_t.p_set["3394"][-3:]) / 3 == n_average.loads_t.p_set["3394"][-1])
print("Generator time data are correctly averaged: ", sum(n.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == n_average.generators_t.p_max_pu["3374 onwind"][-1])

# Calculate per unit impedances and append voltages to lines and shunt impedances
n_average.calculate_dependent_values() 

# optimize the network
n_average.optimize(solver_name='gurobi', solver_options={'Method':2, 'Crossover':0})
#n.optimize(solver_name="gurobi", solver_options={"Method":2})
print("Objective:", n_average.objective)

# define the full file path
network_file_hdf5 = os.path.join(output_folder, "optimized_3h_k(16)-means_network.h5")
# save the network
n_average.export_to_hdf5(network_file_hdf5)
 
n_average.solver = None  # release the solver
gc.collect()  # free model

