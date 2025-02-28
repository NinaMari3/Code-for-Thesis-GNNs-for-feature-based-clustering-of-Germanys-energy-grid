# Kibidis script, I adjusted that the time resolution can be set as an argument
from torch import Tensor
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import pandas as pd
import pypsa

from typing import Optional, Iterable

def node_feature_selection(n: pypsa.components.Network,
                      carriers: Optional[list]=None,
                      include_time: bool=False, 
                      time_resolution: str='2h'
                      ) -> tuple[dict, list]:
    
    df = n.buses

    #Static features
    df = df.rename(columns={"x":"longitude", "y":"latitude"})
    df_features = df[["longitude","latitude"]]
    features = df_features.to_dict('index')

    feature_names = list(df_features.columns)

    if include_time:
        #Time series features
        # finding renewable generators of interest, specified by carrier
        assert carriers is not None, 'Please specify name of carrier for time series data'
        options = carriers
        available_carriers = n.generators['carrier'].unique()
        for carrier in options:
            assert carrier in available_carriers, f"{carrier} not part of {available_carriers}"
        generator_carrier = n.generators[n.generators['carrier'].isin(options)]
        generator_names = generator_carrier.index

        # Get capacity factor time series of generator
        # Time series has 2 hourly resolution (resampled and average over every 2 timesteps)
        generator_capacity_factor = n.generators_t.p_max_pu[generator_names].resample(time_resolution).mean()
        print(time_resolution)
        generator_capacity_factor = generator_capacity_factor.to_dict(orient='list')

        # Generator names and the bus to which they are associated
        generator_bus = n.generators.bus.to_dict()

        bus_names = n.buses.index

        # Create dictionary where keys are buses and values are also dictionaries. Second group of dictionaries contain generators and their associate capacity factor time series
        timesteps = len(list(generator_capacity_factor.values())[0])
        bus_capacity_factors = dict()
        # bus_capacity_factors are zero by default. A key, value pair is created for each generator that is one of the carriers in options
        for b in bus_names:
            bus_capacity_factors[b] = {option:[0]*timesteps for option in options}#{f'{b} solar':[0]*timesteps, f'{b} onwind':[0]*timesteps, f'{b} offwind-ac':[0]*timesteps, f'{b} offwind-dc':[0]*timesteps}

        # bus_capacity_factors are updated based on information from PyPSA-Eur network
        for b in bus_names:
            for key,value in generator_capacity_factor.items():
                if b == generator_bus[key]:
                    carrier = generator_carrier['carrier'][key]
                    bus_capacity_factors[b][carrier] = value 

        # Combine static and time series data
        for bus in features.keys():
            assert bus in bus_capacity_factors.keys(), 'Bus considered is not among buses with renewable capacity factors'
            for generator in bus_capacity_factors[bus].keys():
                #carrier = generator_carrier['carrier'][key]
                features[bus][generator] = bus_capacity_factors[bus][generator] # fixed bug

        # Combine feature names
        feature_names += options 
    
    return features, feature_names

def edge_feature_selection(n: pypsa.components.Network,
                           ) -> dict:
    
    lines = n.lines[['bus0','bus1']]
    resist_react = n.lines[['x','r']] 
    assert (~(n.lines[['x','r']] == 0).all()).all(), 'resistance and reactance are zero. Perhaps run method "calculate_dependant_values()"'
    
    #Calculate admittance
    admittance = 1/(np.sqrt((resist_react.r)**2 + (resist_react.x)**2))
    lines.insert(len(lines.columns), 'admittance', admittance)
    
    admittance = dict()
    for i in lines.index:
        #Need this particular format for key because of format of adding edge attribute to networkx graph constructed from PyPSA network object
        key = (lines.loc[i]['bus0'], lines.loc[i]['bus1'], (lines.index.name, i ))
        admittance[key] = lines.loc[i]['admittance']
    
    return admittance

def pypsa_to_pyg(n: pypsa.components.Network,
                 carriers: Optional[list]=None,
                 buses: Optional[Iterable[int]]=None,
                 include_time: bool=True,
                 verbose: bool=False,
                 time_resolution: str='2h'
        ) -> tuple[Data, Tensor, dict]:
    """
    Converts PyPSA network to PyTorch Geometric graph data

    Parameters:
    -----------
    n: pypsa network data
    carriers: energy carrier such as 'solar'
    buses: buses to be included in pyg object. Used for extracting subgraphs. Default is None which includes all buses
    include_time: include time series data
    verbose: additional detail printed by function

    Returns:
    --------
    pyg_data: the pypsa network in pyg data format
    pyg_adj: the adjacency matrix of network as torch tensor
    mapping: mapping from bus labels to pyg node labels. Bus labels are keys and pyg node labels are values in dict
    """
    
    # Feature selection
    features, feature_names = node_feature_selection(n, carriers, include_time=include_time, time_resolution=time_resolution)
    admittance = edge_feature_selection(n)

    # Create networkx graph
    network = pypsa.Network.graph(n, branch_components=['Line']) #Only consider lines as edges, not links
    nx.set_node_attributes(network, features)
    nx.set_edge_attributes(network, admittance, name='admittance')

    assert network.number_of_edges() == n.lines.shape[0], 'Mismatch with no. edges'
    assert network.number_of_nodes() == n.buses.shape[0], 'Mismatch with no. nodes'

    if buses is not None:
        # Extract sub-graph
        network.remove_nodes_from([n for n in network if n not in set(buses)])
    
    # Mapping from networkx nodes to pyg graph object. Got from PyG function 'from_networkx'
    mapping = dict(zip(network.nodes(), range(network.number_of_nodes())))
    
    #Convert Networkx Graph to PyG Data Object
    pyg_data = pyg_utils.from_networkx(G=network, group_node_attrs=feature_names, group_edge_attrs=['admittance'])
    print(pyg_data.edge_index.size())
    print(pyg_data.edge_index)

    pyg_adj = pyg_utils.to_dense_adj(pyg_data.edge_index).squeeze()
    if verbose:
        print("\n",pyg_data)
        print("Adj size: ",pyg_adj.size(),"\n")
    
    return pyg_data, pyg_adj, mapping, feature_names

def clusters_to_busmap(c: Tensor, 
                       mapping: dict,
                ) -> pd.Series:
    """
    Convert soft cluster assignment matrix to busmap in order to use pypsa

    Parameters:
    -----------
    c: cluster assignment matrix with rows representing nodes and columns the assigned clusters. 
        Entries are probability of being assigned to a cluster.
    mapping: mapping from bus labels to pyg node labels. Bus labels must be the keys and pyg node labels the values in dictionary.
    
    Return:
    -------
    Busmap for PyPSA
    """
    
    busmap = c.max(dim=1)[1].view(-1) # had to add .view(-1) because of dimension problem
    mapping_swapped = dict((value, key) for key, value in mapping.items())
    index = [mapping_swapped[id] for id in range(len(busmap))]

    busmap = pd.Series(busmap, dtype=object)
    busmap = busmap.set_axis(index, axis=0)

    return busmap