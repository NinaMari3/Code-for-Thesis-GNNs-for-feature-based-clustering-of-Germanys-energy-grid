# 100
# script to cluster the network according to different features
# (only coordinates, onwind, offwind-ac, offwind-dc, solar, all)
# 5 runs with different seeds

import pypsa
import torch
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj
from sklearn.preprocessing import MinMaxScaler
import utils
import pickle
import os
import numpy as np

# DMoN
class DMoN(torch.nn.Module):
    def __init__(self, in_channels, clusters, hidden_channels, dropout, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
        np.random.seed(seed)
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True) # weight_initializer='glorot' is implemented in the GCNConv class
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True) #  weight_initializer='glorot' is implemented in the GCNConv class
        self.pool = DMoNPooling(hidden_channels, clusters, dropout=dropout)
    
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

# with lasso regularization (not used)
class DMoN_lasso(torch.nn.Module):
    def __init__(self, in_channels, clusters, hidden_channels, dropout, lasso_lambda = 0.001):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.pool = DMoNPooling(hidden_channels, clusters, dropout=dropout)
        self.lasso_lambda = lasso_lambda  # Regularization strength for Lasso

    def forward(self, x, edge_index, edge_attributes):
        x = x.to(dtype=torch.float)
        edge_attributes = edge_attributes.to(dtype=torch.float)
        x = self.conv1(x, edge_index, edge_weight=edge_attributes).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_attributes).relu()
        adj = to_dense_adj(edge_index, edge_attr=edge_attributes).squeeze()
        adj = adj.to(x.dtype)
        cluster_assignment, pooled_node_feat, adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x, adj)
        lasso_reg = self.lasso_lambda * (torch.norm(self.conv1.lin.weight, p=1)) # L1 norm
        total_loss = 100 * (spectral_loss + cluster_loss) + lasso_reg
        return cluster_assignment, total_loss, 100 * spectral_loss, 100 * cluster_loss, lasso_reg, self.conv1.lin.weight

    
# 1) Data Preparation for Different Configurations
def prepare_data(n, carriers, time_resolution, include_time):

    data, adj, mapping, feature_names = utils.pypsa_to_pyg(n, carriers, verbose=True, include_time=include_time, time_resolution=time_resolution)
    scaler = MinMaxScaler()
    data.x[:, 0:2] = torch.from_numpy(scaler.fit_transform(data.x[:, 0:2]))  # Normalize coordinates
    data.x = data.x.to(torch.float32)

    return data, adj, mapping, feature_names

# Calculate Hidden Channels for different configurations propotional to the number of features
def calculate_hidden_channels(num_features, scaling_factor=0.05, min_hidden=16, max_hidden=1024): # adjust later correctly
    """
    Calculates the number of hidden units (hidden_channels) for large feature sizes
    
    Args:
        num_features: Number of features (F)
        scaling_factor: scaling factor for hidden units
        min_hidden: minimum number of hidden unit, Default is 512.
        max_hidden: Maximum number of hidden units
    
    Returns:
        int: Calculated number of hidden units.
    """
    hidden_channels=512
    #hidden_channels = int(num_features * scaling_factor)
    #hidden_channels = max(min_hidden, min(hidden_channels, max_hidden))
    return hidden_channels

# 2) Training the DMoN Model for Different Configurations

def train_dmon_model(data, carriers, include_time, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)
    print("Training", model_type, "model with", carriers, "carriers and time series data:", include_time)
    """
    Trains the DMoN model with the given configuration.
    
    Args:
        data (torch_geometric.data.Data): The input data for training.
        carriers (list): List of carriers for the configuration.
        include_time (bool): Flag to include time series data.
        model_type (str): Type of the model to train.
        epochs (int): Number of epochs for training.
        lr (float): Learning rate for the optimizer.
        dropout (float): Dropout rate for the model.
        num_clusters (int): Number of clusters for the model.
        lasso_lambda (float): Lasso regularization strength.
        
    Returns:
        dict: A dictionary containing the results of the training process.
    """
    # Initialize and reset model parameters for each configuration
    hidden_units = calculate_hidden_channels(data.x.shape[1]) 
    print("Number of hidden units: ", hidden_units)
    
    
    if model_type == 'DMoN':   
        model = DMoN(
            in_channels=data.x.shape[1],  # Number of input features
            clusters=num_clusters,  # Number of clusters
            hidden_channels=hidden_units,  # Hidden units
            dropout=dropout,
            seed = seed  # Dropout rate
            )
    
    elif model_type == 'DMoN_lasso':
        model = DMoN_lasso(
            in_channels=data.x.shape[1],  # Number of input features
            clusters=num_clusters,  # Number of clusters
            hidden_channels=hidden_units,  # Hidden units
            dropout=dropout,  # Dropout rate
            lasso_lambda=lasso_lambda  # Lasso regularization strength
        )
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    epoch_losses = []
    spectral_losses = []
    cluster_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if model_type == 'DMoN':
            cluster_assignment_softmax, loss, spectral_loss, cluster_loss = model(data.x, data.edge_index, data.edge_attr)
            lasso_reg = None
        
        elif model_type == 'DMoN_lasso':
            cluster_assignment_softmax, loss, spectral_loss, cluster_loss, lasso_reg, weightsGCN1 = model(data.x, data.edge_index, data.edge_attr)
        
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        spectral_losses.append(spectral_loss.item())
        cluster_losses.append(cluster_loss.item())
        
        # Log every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:  
            if model_type == 'DMoN_lasso':
                print(f"  Epoch {epoch + 1}/{epochs}: Loss={loss.item():.4f}, Spectral Loss={spectral_loss.item():.4f}, Cluster Loss={cluster_loss.item():.4f}, Lasso Reg={lasso_reg.item():.4f}")
            elif model_type == 'DMoN':
                print(f"  Epoch {epoch + 1}/{epochs}: Loss={loss.item():.4f}, Spectral Loss={spectral_loss.item():.4f}, Cluster Loss={cluster_loss.item():.4f}")
            
    
    """# Remove batch dimension and convert to busmap for PyPSA
    cluster_assignment_softmax = cluster_assignment_softmax.squeeze(0)
    busmap = utils.clusters_to_busmap(cluster_assignment_softmax, mapping)

    # Cluster the network by busmap
    nc = n.cluster.cluster_by_busmap(busmap)
    """
    
    return {
        "carriers": carriers,
        "include_time": include_time,
        "data": data,
        "model": model, # traied model
        
        "learning_rate": lr,
        "epochs":  epochs,
        "clusters": num_clusters,
        "dropout": dropout,
        "hidden_units": hidden_units,
        "model_type": model_type,
        "lasso_lambda": lasso_lambda,
        "hidden_units": hidden_units,
        
        "epoch_losses": epoch_losses,  
        "spectral_losses": spectral_losses,
        "cluster_losses": cluster_losses,
        "lasso_loss": lasso_reg, 
    }

# 3) test the model
def test_model(n, data, mapping, model, carriers, include_time, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, hidden_units):
        """
        Use the trained model to predict the cluster assignment and evaluate the loss.
        
        Args:
                n (pypsa.Network): The PyPSA network object.
                data (torch_geometric.data.Data): The input data for training.
                mapping (dict): Mapping of nodes to bus IDs.
                model (torch.nn.Module): The trained model.
                carriers (list): List of carriers for the configuration.
                include_time (bool): Flag to include time series data.
                model_type (str): Type of the model to train.
                epochs (int): Number of epochs for training.
                lr (float): Learning rate for the optimizer.
                dropout (float): Dropout rate for the model.
                num_clusters (int): Number of clusters for the model.
                lasso_lambda (float): Lasso regularization strength.
                hidden_units (int): Number of hidden units for the model.
        
        Returns:
        dict: A dictionary containing the results of the testing process.
        """
      
        model.eval() 
        if model_type == 'DMoN':
                 cluster_assignment_softmax, loss, spectral_loss, cluster_loss = model(data.x, data.edge_index, data.edge_attr)
        elif model_type == 'DMoN_lasso':
                        cluster_assignment_softmax, loss, spectral_loss, cluster_loss, lasso_reg, weightsGCN1 = model(data.x, data.edge_index, data.edge_attr)
      
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
                "model_type": model_type,
                "carriers": carriers,
                "include_time": include_time,
                "learning_rate": lr,
                "epochs":  epochs,
                "clusters": num_clusters,
                "dropout": dropout,
                "lasso_lambda": lasso_lambda,
                "hidden_units": hidden_units,
                "mapping": mapping
                }
        n_c = { # for data transfer it has to be separate
                "nc" : nc,
        }
        return tested_model, n_c

# 4) Main Function: Data Preparation and Model Training for All Configurations
def main(n, carriers, include_time, time_resolution, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, seed):
    """
    Trains the DMoN model a specific configurations and returns the model and results after training.
    """
    
    # Prepare data 
    data, adj, mapping, feature_names = prepare_data(n, carriers, time_resolution, include_time)
    
    # Train the DMoN model for this configuration
    trained_model = train_dmon_model(data, carriers, include_time, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, seed)

    return trained_model, mapping

# 5) Run Experiment for Different Configurations
def run_experiment(n, carrier_configs, time_resolution, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, seed):
    """
    Runs the experiment for the given carrier configurations and model type.
    """
    # Train models for each configuration
    training_results = {}
    mappings = {}
    for key, carriers in carrier_configs.items():
        print(carriers)
        print(time_resolution)
        print(model_type)
        
        if key == 'only_coordinates':
            training_results[key], mappings[key] = main(n, carriers, False, time_resolution, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, seed)
        else:
            training_results[key], mappings[key] = main(n, carriers, True, time_resolution, model_type, epochs, lr, dropout, num_clusters, lasso_lambda, seed)
    
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
            train_result['model_type'],
            train_result['epochs'], 
            train_result['learning_rate'], 
            train_result['dropout'],
            train_result['clusters'], 
            train_result['lasso_lambda'], 
            train_result['hidden_units']
        )
        testing_results[key] = tested_model
        testing_results_nc[key] = nc

    return training_results, testing_results, testing_results_nc


# load preprocessed network
path = "/mnt/qb/work/ludwig/lqb072/Thesis4/elec_s.nc" # specify for network on cluster
n = pypsa.Network(path)
n.calculate_dependent_values()


# run clustering with different features

# Hyperparameters for the DMoN model
 
dropout = 0.0
epochs =  15000
lr = 0.0001
cl = 100

# feature selection
carrier_configs = {
        "only_coordinates": [],
        "onwind": ["onwind"],
        "offwind_ac": ["offwind-ac"],
        "offwind_dc": ["offwind-dc"],
        "solar": ["solar"],
        "all": ["onwind", "offwind-ac", "offwind-dc", "solar"]
    }

# Run the experiment
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='2h', model_type='DMoN', epochs=epochs, lr=lr, dropout=dropout, num_clusters=cl, lasso_lambda=None, seed=10)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cf_15000_100_EX1"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cn_15000_100_EX1"
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


# 2nd run
# Run the experiment
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='2h', model_type='DMoN', epochs=epochs, lr=lr, dropout=dropout, num_clusters=cl, lasso_lambda=None, seed=20)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cf_15000_100_EX2"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cn_15000_100_EX2"
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




# 3rd run
# Run the experiment
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='2h', model_type='DMoN', epochs=epochs, lr=lr, dropout=dropout, num_clusters=cl, lasso_lambda=None, seed=30)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cf_15000_100_EX3"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cn_15000_100_EX3"
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

# 2nd run
# Run the experiment
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='2h', model_type='DMoN', epochs=epochs, lr=lr, dropout=dropout, num_clusters=cl, lasso_lambda=None, seed=40)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cf_15000_100_EX4"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cn_15000_100_EX4"
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

# 2nd run
# Run the experiment
training_results, testing_results, clustered_nets = run_experiment(n, carrier_configs, time_resolution='2h', model_type='DMoN', epochs=epochs, lr=lr, dropout=dropout, num_clusters=cl, lasso_lambda=None, seed=50)

# create output folders
output_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cf_15000_100_EX5"
network_directory = "/mnt/qb/work/ludwig/lqb072/Thesis4/lr00001_100/cn_15000_100_EX5"
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
