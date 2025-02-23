# features.py
import numpy as np
import pypsa
import torch
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj
from sklearn.preprocessing import MinMaxScaler
import utils
import pickle
import os
import gurobipy
import gc
import average_time_data

LEARNING_RATE = 0.0001
EPOCHS = 15000
HIDDEN_UNITS = 512

# DMoN
class DMoN(torch.nn.Module):
    def __init__(self, in_channels, clusters, hidden_channels, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        np.random.seed(seed)
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True) # weight_initializer='glorot' is implemented in the GCNConv class
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True) #  weight_initializer='glorot' is implemented in the GCNConv class
        self.pool = DMoNPooling(hidden_channels, clusters)

    def forward(self, x, edge_index, edge_attributes):
        x = x.to(dtype=torch.float)
        edge_attributes = edge_attributes.to(dtype=torch.float)
        x = self.conv1(x, edge_index, edge_weight=edge_attributes).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_attributes).relu()
        adj = to_dense_adj(edge_index, edge_attr=edge_attributes).squeeze()
        adj = adj.to(x.dtype)
        cluster_assignment, pooled_node_feat, adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x, adj)
        loss = 100 * (spectral_loss + cluster_loss)
        return cluster_assignment, loss, 100*spectral_loss, 100*cluster_loss

    
# 1) Data Preparation for Different Configurations
def prepare_data(n, carriers, time_resolution, include_time):

    data, adj, mapping, feature_names = utils.pypsa_to_pyg(n, carriers, verbose=True, include_time=include_time, time_resolution=time_resolution)
    scaler = MinMaxScaler()
    data.x[:, 0:2] = torch.from_numpy(scaler.fit_transform(data.x[:, 0:2]))  # Normalize coordinates
    data.x = data.x.to(torch.float32)

    return data, adj, mapping, feature_names

# 2) Training the DMoN Model for Different Configurations
def train_dmon_model(data, carriers, include_time, num_clusters, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)  # for numpy, if used

    # Initialize and reset model parameters for each configuration
    hidden_units = HIDDEN_UNITS
    print("Number of hidden units: ", hidden_units)
    
    model = DMoN(in_channels=data.x.shape[1],  # Number of input features
            clusters=num_clusters,  # Number of clusters
            hidden_channels=hidden_units,  # Hidden units  
            seed=seed
            )
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    
    # Initialize lists for logging
    epoch_losses = []
    spectral_losses = []
    cluster_losses = []
    #cluster_assignment_softmax = []
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        cluster_assignment_softmax, loss, spectral_loss, cluster_loss = model(data.x, data.edge_index, data.edge_attr)
        loss.backward()
        optimizer.step()
        
        # Log losses
        epoch_losses.append(loss.item())
        spectral_losses.append(spectral_loss.item())
        cluster_losses.append(cluster_loss.item())
        #cluster_assignment_softmax.append(cluster_assignment_softmax.item())
        
        # Log every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:  
            print(f"  Epoch {epoch + 1}/{EPOCHS}: Loss={loss.item():.4f}, Spectral Loss={spectral_loss.item():.4f}, Cluster Loss={cluster_loss.item():.4f}")
    
    return {
        # needed as information
        "carriers": carriers,
        "include_time": include_time,
        "data": data,
        "model": model, # trained model
        
        # hyperparameters 
        "clusters": num_clusters,
        "hidden_units": hidden_units,
        
        # results
        "cluster_assignment_softmax": cluster_assignment_softmax,
        "epoch_losses": epoch_losses,  
        "spectral_losses": spectral_losses,
        "cluster_losses": cluster_losses
    }

# 3) test the model
def test_model(n, data, mapping, model, carriers, include_time, num_clusters):
      
        model.eval() 
        cluster_assignment_softmax, loss, spectral_loss, cluster_loss = model(data.x, data.edge_index, data.edge_attr)
      
        # Remove batch dimension and convert to busmap for PyPSA
        cluster_assignment_softmax = cluster_assignment_softmax.squeeze(0)
        busmap = utils.clusters_to_busmap(cluster_assignment_softmax, mapping)

        # Cluster the network by busmap
        nc = n.cluster.cluster_by_busmap(busmap)
      
        print("Loss: ", loss.item())
        print("Spectral Loss: ", spectral_loss.item())
        print("Cluster Loss: ", cluster_loss.item())
      
        tested_model = {
                "model": model,
                "loss": loss.item(),
                "spectral_loss": spectral_loss.item(),
                "cluster_loss": cluster_loss.item(),
              
                "busmap": busmap,
                "cluster_assignment_softmax": cluster_assignment_softmax,
              
                # needed for plotting
                "carriers": carriers,
                "include_time": include_time,
                "clusters": num_clusters,
                "mapping": mapping
                }
        n_c = { # for data transfer it has to be separate
                "nc" : nc,
        }
        return tested_model, n_c

# 4) Main Function: Data Preparation and Model Training for All Configurations
def main(n, carriers, include_time, time_resolution, num_clusters, seed):
    """
    Trains the DMoN model a specific configurations and returns the model and results after training.
    """
    # Prepare data 
    print(include_time)
    print(carriers)
    print(time_resolution)
    print(num_clusters)
    print(seed)
    data, adj, mapping, feature_names = prepare_data(n, carriers, time_resolution, include_time)
    
    # Train the DMoN model for this configuration
    trained_model = train_dmon_model(data, carriers, include_time, num_clusters, seed)

    return trained_model, mapping

# 5) Run Experiment for Different Configurations
def run_experiment(n, carrier_configs, time_resolution, num_clusters, seed):
    """
    Runs the experiment for the given carrier configurations and model type.
    """
    # Train models for each configuration
    training_results = {}
    mappings = {}
    for key, carriers in carrier_configs.items():
        
        if key == 'only_coordinates':
            training_results[key], mappings[key] = main(n, carriers, False, time_resolution, num_clusters, seed)
        else:
            training_results[key], mappings[key] = main(n, carriers, True, time_resolution, num_clusters, seed)
    
    # Test the trained models
    testing_results = {}
    testing_results_nc = {}
    for key, train_result in training_results.items():
        tested_model, nc = test_model(
            n, 
            train_result['data'], 
            mappings[key], 
            train_result['model'],
            train_result['carriers'], 
            train_result['include_time'], 
            train_result['clusters'], 
        )
        testing_results[key] = tested_model
        testing_results_nc[key] = nc

    return training_results, testing_results, testing_results_nc


# load preprocessed network
path = "/mnt/qb/work/ludwig/lqb072/Thesis4/elec_s.nc" # specify for network on cluster
n = pypsa.Network(path)
n.calculate_dependent_values()

# feature selection
carrier_configs = {
        "all": ["onwind", "offwind-ac", "offwind-dc", "solar"]
    }

# Run the experiment for 16 clusters
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='24h', num_clusters=16, seed = 10)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cf_16_EX1"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cn_16_EX1"
os.makedirs(output_directory, exist_ok=True)
os.makedirs(network_directory, exist_ok=True)

# path
output_file_training = os.path.join(output_directory, "training_data.pkl")
output_file_testing = os.path.join(output_directory, "testing_data.pkl")

# save paras (results dictionary)
with open(output_file_training, 'wb') as f:
    pickle.dump(training_results, f)
with open(output_file_testing, 'wb') as f:
    pickle.dump(testing_results, f)

# save clustered networks
for key, network in clustered_nets.items():
    # Define the full file path
    network_file_hdf5 = os.path.join(network_directory, f"{key}.h5")
    
    # Save the network
    network["nc"].export_to_hdf5(network_file_hdf5)
    print(f"Saved networks in {network_file_hdf5} .")



# Run the experiment for 100
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='24h', num_clusters=100, seed = 10)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cf_100_EX1"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cn_100_EX1"
os.makedirs(output_directory, exist_ok=True)
os.makedirs(network_directory, exist_ok=True)

# path
output_file_training = os.path.join(output_directory, "training_data.pkl")
output_file_testing = os.path.join(output_directory, "testing_data.pkl")

# save paras (results dictionary)
with open(output_file_training, 'wb') as f:
    pickle.dump(training_results, f)
with open(output_file_testing, 'wb') as f:
    pickle.dump(testing_results, f)

# save clustered networks
for key, network in clustered_nets.items():
    # Define the full file path
    network_file_hdf5 = os.path.join(network_directory, f"{key}.h5")
    
    # Save the network
    network["nc"].export_to_hdf5(network_file_hdf5)
    print(f"Saved networks in {network_file_hdf5} .")



# Run the experiment for 100
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='24h', num_clusters=200, seed = 10)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cf_200_EX1"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cn_200_EX1"
os.makedirs(output_directory, exist_ok=True)
os.makedirs(network_directory, exist_ok=True)

# path
output_file_training = os.path.join(output_directory, "training_data.pkl")
output_file_testing = os.path.join(output_directory, "testing_data.pkl")

# save paras (results dictionary)
with open(output_file_training, 'wb') as f:
    pickle.dump(training_results, f)
with open(output_file_testing, 'wb') as f:
    pickle.dump(testing_results, f)

# save clustered networks
for key, network in clustered_nets.items():
    # Define the full file path
    network_file_hdf5 = os.path.join(network_directory, f"{key}.h5")
    
    # Save the network
    network["nc"].export_to_hdf5(network_file_hdf5)
    print(f"Saved networks in {network_file_hdf5} .")


# Optimizaiton of the clustered network

############## 16 clusters ################
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/Thesis4/daily_average/cn_16_EX1"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cn_16_EX1"
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
    

############## 100 clusters ################
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/Thesis4/daily_average/cn_100_EX1"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cn_100_EX1"
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

############## 200 clusters ################
# Define input and output folders
input_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/Thesis4/daily_average/cn_200_EX1"
output_folder = "/mnt/qb/work/ludwig/lqb072/Thesis4/daily_average/cn_200_EX1"
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