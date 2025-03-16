import numpy as np
import pandas as pd

def haversine(lon1, lat1, lon2, lat2):


    # Convert decimal degrees to radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing the haversine formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1),
                           np.multiply(np.cos(lat2),
                               np.power(np.sin(np.divide(dlon, 2)), 2))))

    c = np.multiply(2, np.arcsin(np.sqrt(a)))
    r = 3956

    return (c*r)/1000

def compute_transportation_costs(df, dp, P):
    """ Compute distances and transportation costs. """
    distances = np.vectorize(haversine)(
        np.repeat(df['Longitude'].values, len(dp)),
        np.repeat(df['Latitude'].values, len(dp)),
        np.tile(dp['INTPTLONG'].values, len(df)),
        np.tile(dp['INTPTLAT'].values, len(df))
    )

    distances_df = pd.DataFrame({
        'Warehouse': np.repeat(df['Code'].values, len(dp)),
        'Demand Point': np.tile(dp['ANSICODE'].values, len(df)),
        'Distance': distances
    })

    wgt = {p: np.random.randint(0, 36) for p in P}
    precomputed_values = {p: (0.247847 * wgt[p], 0.132063 * wgt[p], 1.37674 * wgt[p], 0.75604 * wgt[p]) for p in P}

    tr = {}
    for (f, i, distance) in zip(distances_df['Warehouse'], distances_df['Demand Point'], distances_df['Distance']):
        for p in P:
            w1, w2, w3, w4 = precomputed_values[p]
            tr[f, i, p] = w1 * distance - 0.227878 * distance + w2 + 7.553833 if distance <= 1 else w3 * distance + 4.02487 * distance + w4 + 11.10890
    
    return tr