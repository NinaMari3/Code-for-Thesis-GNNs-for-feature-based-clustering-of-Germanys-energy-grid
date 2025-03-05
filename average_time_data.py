# average time data in a pypsa network for a given time window to be able to solve / optimize the network 

import pypsa
import pandas as pd
import numpy as np

def average_snapshots(network, time_window=3):
    # loop through all attributes 
    for attr in dir(network):
        # check if the attribute ends with '_t' (means time series data)
        if attr.endswith("_t"):
            time_series_data = getattr(network, attr)  
            #print(attr)

            if isinstance(time_series_data, dict):  
                for key, df in time_series_data.items():
                    if isinstance(df, (pd.DataFrame, pd.Series)):
                        # Resample and get mean
                        time_series_data[key] = df.resample(f'{time_window}h').mean()
            elif isinstance(time_series_data, pd.DataFrame):
                setattr(network, attr, time_series_data.resample(f'{time_window}h').mean())

    network.snapshots = network.snapshots[::time_window]
    network.snapshot_weightings *= time_window # The weighting of the snapshots (e.g. how many hours they represent

    
# test if the average is correct 
#print("Loads are correctly averaged: ", sum(n.loads_t.p_set["3394"][-3:]) / 3 == n_average.loads_t.p_set["3394"][-1])
#print("Generator time data are correctly averaged: ", sum(n.generators_t.p_max_pu["3374 onwind"][-3:]) / 3 == n_average.generators_t.p_max_pu["3374 onwind"][-1])
