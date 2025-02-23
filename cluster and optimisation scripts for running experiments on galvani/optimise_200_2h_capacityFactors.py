# optimization of clustered networks

import pypsa
import os
import gurobipy
import gc
import average_time_data

############## 1
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_200/cn_15000_200_EX1"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/optim_200cl/optimized_cl_200_EX1"
os.makedirs(output_folder, exist_ok=True)

# List all HDF5 files in the folder
network_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

# Initialize a dictionary to store the loaded networks with their names
networks = {}

# Reload each network from the HDF5 files
for network_file in network_files:
    # Generate the full file path
    network_file_hdf5 = os.path.join(input_folder, network_file)
    
    # Reload the network
    network = pypsa.Network()
    network.import_from_hdf5(network_file_hdf5)
    
    # Value befor averaging 
    print("Loads before averaged (average of last 3 of node 3394): ", sum(network.loads_t.p_set["3394"][-3:]) / 3)
    print("Generator time data before averaged (average of last 3 of generator 3374 onwind): ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3)
    
    # Average the time data
    average_time_data.average_snapshots(network, time_window=3)
    
    # Vlaue after averaging
    #print("Loads after averaged (last of node 3394): ", network.loads_t.p_set["3394"][-1])
    # print("Generator time data after averaged (last of generator 3374 onwind): ", network.generators_t.p_max_pu["3374 onwind"][-1])
    #print("Loads are correctly averaged: ", sum(network.loads_t.p_set["3394"][-3:]) / 3 == network.loads_t.p_set["3394"][-1])
    #print("Generator time data are correctly averaged: ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == network.generators_t.p_max_pu["3374 onwind"][-1])

    # Use the file name (without .h5) as the key in the dictionary
    network_name = os.path.splitext(network_file)[0]
    networks[network_name] = network
    
    print(f"Loaded network '{network_name}' from {network_file_hdf5}")

# Optimize and save each network
for name, net in networks.items():
    #n = result["nc"]
    net.optimize(solver_name='gurobi', solver_options={'Method':2, 'Crossover':0})
    optimized_network_path = os.path.join(output_folder, f"{name}_optimized.h5")
    net.export_to_hdf5(optimized_network_path)
    print(f"Optimized and saved network '{name}' to {optimized_network_path}")

    gc.collect()  # set garbage collector to free memory

############## 2
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_200/cn_15000_200_EX2"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/optim_200cl/optimized_cl_200_EX2"
os.makedirs(output_folder, exist_ok=True)

# List all HDF5 files in the folder
network_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

# Initialize a dictionary to store the loaded networks with their names
networks = {}

# Reload each network from the HDF5 files
for network_file in network_files:
    # Generate the full file path
    network_file_hdf5 = os.path.join(input_folder, network_file)
    
    # Reload the network
    network = pypsa.Network()
    network.import_from_hdf5(network_file_hdf5)
    
    # Value befor averaging 
    print("Loads before averaged (average of last 3 of node 3394): ", sum(network.loads_t.p_set["3394"][-3:]) / 3)
    print("Generator time data before averaged (average of last 3 of generator 3374 onwind): ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3)
    
    # Average the time data
    average_time_data.average_snapshots(network, time_window=3)
    
    # Vlaue after averaging
    #print("Loads after averaged (last of node 3394): ", network.loads_t.p_set["3394"][-1])
    # print("Generator time data after averaged (last of generator 3374 onwind): ", network.generators_t.p_max_pu["3374 onwind"][-1])
    #print("Loads are correctly averaged: ", sum(network.loads_t.p_set["3394"][-3:]) / 3 == network.loads_t.p_set["3394"][-1])
    #print("Generator time data are correctly averaged: ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == network.generators_t.p_max_pu["3374 onwind"][-1])

    # Use the file name (without .h5) as the key in the dictionary
    network_name = os.path.splitext(network_file)[0]
    networks[network_name] = network
    
    print(f"Loaded network '{network_name}' from {network_file_hdf5}")

# Optimize and save each network
for name, net in networks.items():
    #n = result["nc"]
    net.optimize(solver_name='gurobi', solver_options={'Method':2, 'Crossover':0})
    optimized_network_path = os.path.join(output_folder, f"{name}_optimized.h5")
    net.export_to_hdf5(optimized_network_path)
    print(f"Optimized and saved network '{name}' to {optimized_network_path}")

    gc.collect()  # set garbage collector to free memory
    

############## 3 
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_200/cn_15000_200_EX3"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/optim_200cl/optimized_cl_200_EX3"
os.makedirs(output_folder, exist_ok=True)

# List all HDF5 files in the folder
network_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

# Initialize a dictionary to store the loaded networks with their names
networks = {}

# Reload each network from the HDF5 files
for network_file in network_files:
    # Generate the full file path
    network_file_hdf5 = os.path.join(input_folder, network_file)
    
    # Reload the network
    network = pypsa.Network()
    network.import_from_hdf5(network_file_hdf5)
    
    # Value befor averaging 
    print("Loads before averaged (average of last 3 of node 3394): ", sum(network.loads_t.p_set["3394"][-3:]) / 3)
    print("Generator time data before averaged (average of last 3 of generator 3374 onwind): ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3)
    
    # Average the time data
    average_time_data.average_snapshots(network, time_window=3)
    
    # Vlaue after averaging
    #print("Loads after averaged (last of node 3394): ", network.loads_t.p_set["3394"][-1])
    # print("Generator time data after averaged (last of generator 3374 onwind): ", network.generators_t.p_max_pu["3374 onwind"][-1])
    #print("Loads are correctly averaged: ", sum(network.loads_t.p_set["3394"][-3:]) / 3 == network.loads_t.p_set["3394"][-1])
    #print("Generator time data are correctly averaged: ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == network.generators_t.p_max_pu["3374 onwind"][-1])

    # Use the file name (without .h5) as the key in the dictionary
    network_name = os.path.splitext(network_file)[0]
    networks[network_name] = network
    
    print(f"Loaded network '{network_name}' from {network_file_hdf5}")

# Optimize and save each network
for name, net in networks.items():
    #n = result["nc"]
    net.optimize(solver_name='gurobi', solver_options={'Method':2, 'Crossover':0})
    optimized_network_path = os.path.join(output_folder, f"{name}_optimized.h5")
    net.export_to_hdf5(optimized_network_path)
    print(f"Optimized and saved network '{name}' to {optimized_network_path}")

    gc.collect()  # set garbage collector to free memory


############## 4
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_200/cn_15000_200_EX4"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/optim_200cl/optimized_cl_200_EX4"
os.makedirs(output_folder, exist_ok=True)

# List all HDF5 files in the folder
network_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

# Initialize a dictionary to store the loaded networks with their names
networks = {}

# Reload each network from the HDF5 files
for network_file in network_files:
    # Generate the full file path
    network_file_hdf5 = os.path.join(input_folder, network_file)
    
    # Reload the network
    network = pypsa.Network()
    network.import_from_hdf5(network_file_hdf5)
    
    # Value befor averaging 
    print("Loads before averaged (average of last 3 of node 3394): ", sum(network.loads_t.p_set["3394"][-3:]) / 3)
    print("Generator time data before averaged (average of last 3 of generator 3374 onwind): ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3)
    
    # Average the time data
    average_time_data.average_snapshots(network, time_window=3)
    
    # Vlaue after averaging
    #print("Loads after averaged (last of node 3394): ", network.loads_t.p_set["3394"][-1])
    # print("Generator time data after averaged (last of generator 3374 onwind): ", network.generators_t.p_max_pu["3374 onwind"][-1])
    #print("Loads are correctly averaged: ", sum(network.loads_t.p_set["3394"][-3:]) / 3 == network.loads_t.p_set["3394"][-1])
    #print("Generator time data are correctly averaged: ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == network.generators_t.p_max_pu["3374 onwind"][-1])

    # Use the file name (without .h5) as the key in the dictionary
    network_name = os.path.splitext(network_file)[0]
    networks[network_name] = network
    
    print(f"Loaded network '{network_name}' from {network_file_hdf5}")

# Optimize and save each network
for name, net in networks.items():
    #n = result["nc"]
    net.optimize(solver_name='gurobi', solver_options={'Method':2, 'Crossover':0})
    optimized_network_path = os.path.join(output_folder, f"{name}_optimized.h5")
    net.export_to_hdf5(optimized_network_path)
    print(f"Optimized and saved network '{name}' to {optimized_network_path}")

    gc.collect()  # set garbage collector to free memory

############## 5
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_200/cn_15000_200_EX5"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/optim_200cl/optimized_cl_200_EX5"
os.makedirs(output_folder, exist_ok=True)

# List all HDF5 files in the folder
network_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

# Initialize a dictionary to store the loaded networks with their names
networks = {}

# Reload each network from the HDF5 files
for network_file in network_files:
    # Generate the full file path
    network_file_hdf5 = os.path.join(input_folder, network_file)
    
    # Reload the network
    network = pypsa.Network()
    network.import_from_hdf5(network_file_hdf5)
    
    # Value befor averaging 
    print("Loads before averaged (average of last 3 of node 3394): ", sum(network.loads_t.p_set["3394"][-3:]) / 3)
    print("Generator time data before averaged (average of last 3 of generator 3374 onwind): ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3)
    
    # Average the time data
    average_time_data.average_snapshots(network, time_window=3)
    
    # Vlaue after averaging
    #print("Loads after averaged (last of node 3394): ", network.loads_t.p_set["3394"][-1])
    # print("Generator time data after averaged (last of generator 3374 onwind): ", network.generators_t.p_max_pu["3374 onwind"][-1])
    #print("Loads are correctly averaged: ", sum(network.loads_t.p_set["3394"][-3:]) / 3 == network.loads_t.p_set["3394"][-1])
    #print("Generator time data are correctly averaged: ", sum(network.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == network.generators_t.p_max_pu["3374 onwind"][-1])

    # Use the file name (without .h5) as the key in the dictionary
    network_name = os.path.splitext(network_file)[0]
    networks[network_name] = network
    
    print(f"Loaded network '{network_name}' from {network_file_hdf5}")

# Optimize and save each network
for name, net in networks.items():
    #n = result["nc"]
    net.optimize(solver_name='gurobi', solver_options={'Method':2, 'Crossover':0})
    optimized_network_path = os.path.join(output_folder, f"{name}_optimized.h5")
    net.export_to_hdf5(optimized_network_path)
    print(f"Optimized and saved network '{name}' to {optimized_network_path}")

    gc.collect()  # set garbage collector to free memory

