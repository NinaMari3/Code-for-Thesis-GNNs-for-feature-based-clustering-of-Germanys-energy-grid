# average time data in a pypsa network for a given time window to be able to solve / optimize the network 

import pypsa
import pandas as pd
import numpy as np

def average_snapshots(network, time_window=3):
    """
    Resample all time series data (_t-attributes) of a PyPSA network and calculate the average for a given time window
    """
    # Iterate through all attributes of the network
    for attr in dir(network):
        # Check if the attribute ends with '_t' (indicating time series data)
        if attr.endswith("_t"):
            time_series_data = getattr(network, attr)  # Access the time series attribute
            #print(attr)
            
            # If the attribute is a dictionary, resample each DataFrame/Series
            if isinstance(time_series_data, dict):  
                for key, df in time_series_data.items():
                    if isinstance(df, (pd.DataFrame, pd.Series)):
                        # Resample and compute the mean
                        time_series_data[key] = df.resample(f'{time_window}h').mean()
            
            # If the attribute is a DataFrame, resample directly
            elif isinstance(time_series_data, pd.DataFrame):
                setattr(network, attr, time_series_data.resample(f'{time_window}h').mean())

    # Adjust the network snapshots to reflect the new time window
    network.snapshots = network.snapshots[::time_window]
    network.snapshot_weightings *= time_window # The weighting of the snapshots (e.g. how many hours they represent

    
# test if the average is correct 
#print("Loads are correctly averaged: ", sum(n.loads_t.p_set["3394"][-3:]) / 3 == n_average.loads_t.p_set["3394"][-1])
#print("Generator time data are correctly averaged: ", sum(n.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == n_average.generators_t.p_max_pu["3374 onwind"][-1])
